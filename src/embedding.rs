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

/// Trait for pluggable embedding providers.
/// Implement this to add support for custom embedding APIs (e.g., Ollama, Cohere, local models).
#[async_trait::async_trait]
pub trait EmbeddingProvider: Send + Sync {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f64>>, EmbeddingError>;
    fn name(&self) -> &str;
    fn dimensions(&self) -> Option<usize> {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, Serialize, Deserialize)]
pub enum EmbeddingProviderKind {
    Openai,
    Gemini,
}

impl std::fmt::Display for EmbeddingProviderKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbeddingProviderKind::Openai => write!(f, "openai"),
            EmbeddingProviderKind::Gemini => write!(f, "gemini"),
        }
    }
}

/// Create a boxed embedding provider from kind + optional key/model.
pub fn create_embedding_provider(
    kind: EmbeddingProviderKind,
    api_key: Option<String>,
    model: Option<String>,
) -> Result<Box<dyn EmbeddingProvider>, EmbeddingError> {
    match kind {
        EmbeddingProviderKind::Openai => Ok(Box::new(OpenAiProvider::new(api_key, model)?)),
        EmbeddingProviderKind::Gemini => Ok(Box::new(GeminiProvider::new(api_key, model)?)),
    }
}

// --- OpenAI Provider ---

pub struct OpenAiProvider {
    client: Client,
    api_key: String,
    model: String,
}

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

impl OpenAiProvider {
    pub fn new(api_key: Option<String>, model: Option<String>) -> Result<Self, EmbeddingError> {
        let api_key = api_key
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .ok_or_else(|| EmbeddingError::MissingApiKey {
                env_var: "OPENAI_API_KEY".to_string(),
            })?;
        Ok(Self {
            client: Client::builder().timeout(Duration::from_secs(60)).build()?,
            api_key,
            model: model.unwrap_or_else(|| "text-embedding-3-small".to_string()),
        })
    }
}

#[async_trait::async_trait]
impl EmbeddingProvider for OpenAiProvider {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f64>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
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

    fn name(&self) -> &str {
        "openai"
    }

    fn dimensions(&self) -> Option<usize> {
        Some(1536)
    }
}

// --- Gemini Provider ---

pub struct GeminiProvider {
    client: Client,
    api_key: String,
    model: String,
}

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

impl GeminiProvider {
    pub fn new(api_key: Option<String>, model: Option<String>) -> Result<Self, EmbeddingError> {
        let api_key = api_key
            .or_else(|| std::env::var("GEMINI_API_KEY").ok())
            .ok_or_else(|| EmbeddingError::MissingApiKey {
                env_var: "GEMINI_API_KEY".to_string(),
            })?;
        Ok(Self {
            client: Client::builder().timeout(Duration::from_secs(60)).build()?,
            api_key,
            model: model.unwrap_or_else(|| "models/text-embedding-004".to_string()),
        })
    }
}

#[async_trait::async_trait]
impl EmbeddingProvider for GeminiProvider {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f64>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
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

    fn name(&self) -> &str {
        "gemini"
    }

    fn dimensions(&self) -> Option<usize> {
        Some(768)
    }
}
