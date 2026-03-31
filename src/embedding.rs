use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("API error: {status} {message}")]
    Api { status: u16, message: String },
    #[error("Missing API key: set {env_var}")]
    MissingApiKey { env_var: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, Serialize, Deserialize)]
pub enum EmbeddingProvider {
    Openai,
    Gemini,
}

impl std::fmt::Display for EmbeddingProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbeddingProvider::Openai => write!(f, "openai"),
            EmbeddingProvider::Gemini => write!(f, "gemini"),
        }
    }
}

pub struct EmbeddingClient {
    client: Client,
    provider: EmbeddingProvider,
    api_key: String,
    model: String,
}

// --- OpenAI types ---

#[derive(Serialize)]
struct OpenAiEmbeddingRequest {
    input: Vec<String>,
    model: String,
}

#[derive(Deserialize)]
struct OpenAiEmbeddingResponse {
    data: Vec<OpenAiEmbeddingData>,
}

#[derive(Deserialize)]
struct OpenAiEmbeddingData {
    embedding: Vec<f64>,
}

#[derive(Deserialize)]
struct OpenAiErrorResponse {
    error: OpenAiErrorDetail,
}

#[derive(Deserialize)]
struct OpenAiErrorDetail {
    message: String,
}

// --- Gemini types ---

#[derive(Serialize)]
struct GeminiEmbedRequest {
    requests: Vec<GeminiEmbedContentRequest>,
}

#[derive(Serialize)]
struct GeminiEmbedContentRequest {
    model: String,
    content: GeminiContent,
}

#[derive(Serialize)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
}

#[derive(Serialize)]
struct GeminiPart {
    text: String,
}

#[derive(Deserialize)]
struct GeminiBatchEmbedResponse {
    embeddings: Vec<GeminiEmbedding>,
}

#[derive(Deserialize)]
struct GeminiEmbedding {
    values: Vec<f64>,
}

#[derive(Deserialize)]
struct GeminiErrorResponse {
    error: GeminiErrorDetail,
}

#[derive(Deserialize)]
struct GeminiErrorDetail {
    message: String,
}

impl EmbeddingClient {
    pub fn new(
        provider: EmbeddingProvider,
        api_key: Option<String>,
        model: Option<String>,
    ) -> Result<Self, EmbeddingError> {
        let (env_var, default_model) = match provider {
            EmbeddingProvider::Openai => ("OPENAI_API_KEY", "text-embedding-3-small"),
            EmbeddingProvider::Gemini => ("GEMINI_API_KEY", "models/text-embedding-004"),
        };

        let api_key = api_key
            .or_else(|| std::env::var(env_var).ok())
            .ok_or_else(|| EmbeddingError::MissingApiKey {
                env_var: env_var.to_string(),
            })?;

        let model = model.unwrap_or_else(|| default_model.to_string());

        Ok(Self {
            client: Client::builder().timeout(Duration::from_secs(60)).build()?,
            provider,
            api_key,
            model,
        })
    }

    pub async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f64>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        match self.provider {
            EmbeddingProvider::Openai => self.embed_openai(texts).await,
            EmbeddingProvider::Gemini => self.embed_gemini(texts).await,
        }
    }

    async fn embed_openai(&self, texts: &[String]) -> Result<Vec<Vec<f64>>, EmbeddingError> {
        let body = OpenAiEmbeddingRequest {
            input: texts.to_vec(),
            model: self.model.clone(),
        };

        let resp = self
            .client
            .post("https://api.openai.com/v1/embeddings")
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            let message = serde_json::from_str::<OpenAiErrorResponse>(&text)
                .map(|e| e.error.message)
                .unwrap_or(text);
            return Err(EmbeddingError::Api {
                status: status.as_u16(),
                message,
            });
        }

        let data: OpenAiEmbeddingResponse = resp.json().await?;
        Ok(data.data.into_iter().map(|d| d.embedding).collect())
    }

    async fn embed_gemini(&self, texts: &[String]) -> Result<Vec<Vec<f64>>, EmbeddingError> {
        let requests: Vec<GeminiEmbedContentRequest> = texts
            .iter()
            .map(|t| GeminiEmbedContentRequest {
                model: self.model.clone(),
                content: GeminiContent {
                    parts: vec![GeminiPart { text: t.clone() }],
                },
            })
            .collect();

        let body = GeminiEmbedRequest { requests };

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/{}:batchEmbedContents?key={}",
            self.model, self.api_key
        );

        let resp = self.client.post(&url).json(&body).send().await?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            let message = serde_json::from_str::<GeminiErrorResponse>(&text)
                .map(|e| e.error.message)
                .unwrap_or(text);
            return Err(EmbeddingError::Api {
                status: status.as_u16(),
                message,
            });
        }

        let data: GeminiBatchEmbedResponse = resp.json().await?;
        Ok(data.embeddings.into_iter().map(|e| e.values).collect())
    }
}
