use burn::config::Config;
use burn::module::Module;
use burn::nn::LayerNormConfig;
use burn::tensor::Tensor;
use burn::{nn::LayerNorm, tensor::backend::Backend};

use super::mlp::{Mlp, MlpConfig};
use super::multi_head_attention::{MultiHeadAttention, MultiHeadAttentionConfig};

#[derive(Debug, Module)]
pub struct Encoder<B: Backend> {
    mha: MultiHeadAttention<B>,
    ln1: LayerNorm<B>,
    ln2: LayerNorm<B>,
    mlp: Mlp<B>,
}

#[derive(Debug, Config)]
pub struct EncoderConfig {
    embedding_dim: usize,
    num_heads: usize,
    hidden_dim: usize,
    dropout: f64,
}

impl EncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Encoder<B> {
        let head_dim = self.embedding_dim / self.num_heads;
        Encoder {
            mha: MultiHeadAttentionConfig::new(self.num_heads, self.embedding_dim, head_dim)
                .init(device),
            ln1: LayerNormConfig::new(self.embedding_dim).init(device),
            ln2: LayerNormConfig::new(self.embedding_dim).init(device),
            mlp: MlpConfig::new(self.embedding_dim, self.hidden_dim).init(device),
        }
    }
}

impl<B: Backend> Encoder<B> {
    pub fn forward(&self, inputs: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut x_ = inputs.clone();
        let mut x = self.mha.forward(inputs);
        x = self.ln1.forward(x) + x_;
        x_ = x.clone();
        x = self.mlp.forward(x);
        self.ln2.forward(x) + x_
    }
}
