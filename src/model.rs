use burn::{
    config::Config, 
    module::Module, 
    nn::{LayerNorm, LayerNormConfig}, 
    tensor::{backend::Backend, Tensor}
};
use super::layers::{encoder, patch_embedding, position_embedding, pooler};

#[derive(Debug, Module)]
pub struct ViTModel<B: Backend> {
    patch_embedding: patch_embedding::PatchEmbedding<B>,
    positional_embedding: position_embedding::PositionalEmbedding<B>,
    encoders: Vec<encoder::Encoder<B>>,
    ln1: LayerNorm<B>,
    pooler: pooler::Pooler<B>,
}

#[derive(Debug, Config)]
pub struct ViTConfig {
    embedding_dim: usize,
    patch_size: usize,
    in_channels: usize,
    max_token_length: usize,
    hidden_dim: usize,
    num_heads: usize,
    dropout: f64,
    num_encoder_layers: usize,   
}

impl ViTConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ViTModel<B> {
        let mut encoders: Vec<encoder::Encoder<B>> = vec![];

        for _ in 0..self.num_encoder_layers {
            encoders.push(
                encoder::EncoderConfig::new(self.embedding_dim, self.num_heads, self.hidden_dim, self.dropout)
                .init(device)
            );
        }

        ViTModel { 
            patch_embedding: patch_embedding::PatchEmbeddingConfig::new(self.patch_size, self.in_channels).init(device),
            positional_embedding: position_embedding::PositionalEmbeddingConfig::new().init(device),
            encoders: encoders,
            ln1: LayerNormConfig::new(self.embedding_dim).init(device),
            pooler: pooler::PoolerConfig::new(self.embedding_dim, self.embedding_dim).init(device),
        }
    }
}

impl<B: Backend> ViTModel<B> {
    pub fn forward(&self, inputs: Tensor<B, 4>) -> Tensor<B, 3> {
        // inputs: [B, H, W, C]
        // output: [B, S, E]
        let patch_outputs = self.patch_embedding.forward(inputs);
        let mut x = self.positional_embedding.forward(patch_outputs);

        for lyr in self.encoders.clone() {
            x = lyr.forward(x);
        };

        return x
    }
}
