use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[cfg(feature = "fastembed")]
use std::sync::{Arc, Mutex};

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("API error: {status} {message}")]
    Api { status: u16, message: String },
    #[error("Missing API key: set {env_var}")]
    MissingApiKey { env_var: String },
    #[error("Feature not enabled: rebuild with `{feature}`")]
    FeatureDisabled { feature: String },
    #[error("Embedding provider error: {message}")]
    Provider { message: String },
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
#[serde(rename_all = "lowercase")]
pub enum EmbeddingProviderKind {
    Openai,
    Gemini,
    Ollama,
    Fastembed,
}

impl std::fmt::Display for EmbeddingProviderKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbeddingProviderKind::Openai => write!(f, "openai"),
            EmbeddingProviderKind::Gemini => write!(f, "gemini"),
            EmbeddingProviderKind::Ollama => write!(f, "ollama"),
            EmbeddingProviderKind::Fastembed => write!(f, "fastembed"),
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
        EmbeddingProviderKind::Ollama => Ok(Box::new(OllamaProvider::new(api_key, model)?)),
        #[cfg(feature = "fastembed")]
        EmbeddingProviderKind::Fastembed => Ok(Box::new(FastembedProvider::new(api_key, model)?)),
        #[cfg(not(feature = "fastembed"))]
        EmbeddingProviderKind::Fastembed => Err(EmbeddingError::FeatureDisabled {
            feature: "fastembed".to_string(),
        }),
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

// --- Ollama Provider (local LLM) ---

/// Embedding provider for Ollama (local LLM server).
/// Default model: `nomic-embed-text`. No API key required.
/// Set `OLLAMA_HOST` to override the default URL (http://localhost:11434).
pub struct OllamaProvider {
    client: Client,
    base_url: String,
    model: String,
}

#[derive(Serialize)]
struct OllamaEmbedRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize)]
struct OllamaEmbedResponse {
    embeddings: Vec<Vec<f64>>,
}

impl OllamaProvider {
    pub fn new(_api_key: Option<String>, model: Option<String>) -> Result<Self, EmbeddingError> {
        let base_url =
            std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string());
        Ok(Self {
            client: Client::builder()
                .timeout(Duration::from_secs(120))
                .build()?,
            base_url,
            model: model.unwrap_or_else(|| "nomic-embed-text".to_string()),
        })
    }
}

#[async_trait::async_trait]
impl EmbeddingProvider for OllamaProvider {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f64>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        let body = OllamaEmbedRequest {
            model: self.model.clone(),
            input: texts.to_vec(),
        };
        let url = format!("{}/api/embed", self.base_url);
        let resp = self.client.post(&url).json(&body).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(EmbeddingError::Api {
                status: status.as_u16(),
                message: text,
            });
        }
        let data: OllamaEmbedResponse = resp.json().await?;
        Ok(data.embeddings)
    }

    fn name(&self) -> &str {
        "ollama"
    }
}

// --- FastEmbed Provider (serverless local embedding) ---

#[cfg(feature = "fastembed")]
pub struct FastembedProvider {
    model: Arc<Mutex<fastembed::TextEmbedding>>,
    dimensions: usize,
}

#[cfg(feature = "fastembed")]
impl FastembedProvider {
    pub fn new(_api_key: Option<String>, model: Option<String>) -> Result<Self, EmbeddingError> {
        use fastembed::{InitOptions, TextEmbedding};

        let model_name = model.unwrap_or_else(|| "BAAI/bge-small-en-v1.5".to_string());
        let embedding_model = parse_fastembed_model(&model_name)?;
        let dimensions = TextEmbedding::get_model_info(&embedding_model)
            .map_err(|error| EmbeddingError::Provider {
                message: error.to_string(),
            })?
            .dim;
        let text_embedding = TextEmbedding::try_new(
            InitOptions::new(embedding_model).with_show_download_progress(false),
        )
        .map_err(|error| EmbeddingError::Provider {
            message: error.to_string(),
        })?;

        Ok(Self {
            model: Arc::new(Mutex::new(text_embedding)),
            dimensions,
        })
    }
}

#[cfg(feature = "fastembed")]
fn parse_fastembed_model(model_name: &str) -> Result<fastembed::EmbeddingModel, EmbeddingError> {
    use fastembed::EmbeddingModel;

    let canonical = model_name.trim();
    let alias = match canonical {
        "BAAI/bge-small-en-v1.5" => Some(EmbeddingModel::BGESmallENV15),
        "BAAI/bge-base-en-v1.5" => Some(EmbeddingModel::BGEBaseENV15),
        "BAAI/bge-large-en-v1.5" => Some(EmbeddingModel::BGELargeENV15),
        "sentence-transformers/all-MiniLM-L6-v2" => Some(EmbeddingModel::AllMiniLML6V2),
        "sentence-transformers/all-MiniLM-L12-v2" => Some(EmbeddingModel::AllMiniLML12V2),
        "nomic-ai/nomic-embed-text-v1" => Some(EmbeddingModel::NomicEmbedTextV1),
        "nomic-ai/nomic-embed-text-v1.5" => Some(EmbeddingModel::NomicEmbedTextV15),
        _ => None,
    };

    if let Some(model) = alias {
        Ok(model)
    } else {
        canonical
            .parse::<EmbeddingModel>()
            .map_err(|message| EmbeddingError::Provider {
                message: message.to_string(),
            })
    }
}

#[cfg(feature = "fastembed")]
#[async_trait::async_trait]
impl EmbeddingProvider for FastembedProvider {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f64>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let texts = texts.to_vec();
        let model = Arc::clone(&self.model);
        let embeddings = tokio::task::spawn_blocking(move || {
            let model = model.lock().map_err(|error| EmbeddingError::Provider {
                message: format!("fastembed model lock poisoned: {error}"),
            })?;
            model
                .embed(texts, None)
                .map_err(|error| EmbeddingError::Provider {
                    message: error.to_string(),
                })
        })
        .await
        .map_err(|error| EmbeddingError::Provider {
            message: format!("fastembed worker failed: {error}"),
        })??;

        Ok(embeddings
            .into_iter()
            .map(|embedding| embedding.into_iter().map(f64::from).collect())
            .collect())
    }

    fn name(&self) -> &str {
        "fastembed"
    }

    fn dimensions(&self) -> Option<usize> {
        Some(self.dimensions)
    }
}

#[cfg(all(test, feature = "fastembed"))]
mod fastembed_tests {
    use super::*;

    #[tokio::test]
    async fn test_fastembed_provider_embeds_text() {
        let provider = FastembedProvider::new(None, None).unwrap();
        assert_eq!(provider.name(), "fastembed");
        assert_eq!(provider.dimensions(), Some(384));

        let embeddings = provider
            .embed(&[
                "query: rust".to_string(),
                "passage: systems programming".to_string(),
            ])
            .await
            .unwrap();

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 384);
    }
}
