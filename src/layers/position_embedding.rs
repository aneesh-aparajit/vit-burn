use std::{ops::Range, vec};

use burn::{
    nn::{Dropout, DropoutConfig}, prelude::*, tensor::backend::Backend
};

#[derive(Debug, Module)]
pub struct PositionalEmbedding<B: Backend> {
    positional_embedding_matrix: Tensor::<B, 2, Float>,
    dropout: Dropout
}

#[derive(Debug, Config)]
pub struct PositionalEmbeddingConfig {
    #[config(default = 768)]
    embedding_dim: usize,
    #[config(default = 512)]
    max_token_length: usize,
    #[config(default = "0.0")]
    dropout: f64
}


impl<B: Backend> PositionalEmbedding<B> {
    pub fn new(embedding_dim: usize, max_token_length: usize, dropout: f64, device: &B::Device) -> PositionalEmbedding<B> {
        let position_ix: Tensor<B, 1, Float> = Tensor::arange(Range { start: 0, end: max_token_length as i64 }, device).float();
        let denominator: Tensor<B, 1, Float> = Tensor::ones_like(&position_ix).mul_scalar(10000.0);
        let ix = 0;

        let even_embeds = position_ix.clone().div(denominator.clone()
            .powf_scalar(2.0*(ix as f32)/(embedding_dim as f32))).sin()
            .clone().reshape([max_token_length, 1]);

        let odd_embeds = position_ix.clone().div(denominator.clone()
            .powf_scalar(2.0*(ix as f32)/(embedding_dim as f32))).cos()
            .clone().reshape([max_token_length, 1]);

        let mut positional_embedding = Tensor::cat(vec![even_embeds, odd_embeds], 1);
        for ix  in 1..embedding_dim/2 {
            let even_embeds = position_ix.clone().div(denominator.clone()
                .powf_scalar(2.0*(ix as f32)/(embedding_dim as f32))).sin()
                .clone().reshape([max_token_length, 1]);

            let odd_embeds = position_ix.clone().div(denominator.clone()
                .powf_scalar(2.0*(ix as f32)/(embedding_dim as f32))).cos().
                clone().reshape([max_token_length, 1]);
            
            positional_embedding = Tensor::cat(vec![positional_embedding, even_embeds], 1);
            positional_embedding = Tensor::cat(vec![positional_embedding, odd_embeds], 1);
        };

        PositionalEmbedding { 
            positional_embedding_matrix: positional_embedding, 
            dropout: DropoutConfig::new(dropout).init() 
        }
    }
}

impl PositionalEmbeddingConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PositionalEmbedding<B> {
        PositionalEmbedding::new(self.embedding_dim, self.max_token_length, self.dropout, device)
    }
}

impl<B: Backend> PositionalEmbedding<B> {
    pub fn forward(&self, inputs: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, embedding_dim] = inputs.dims();

        let mut x = inputs.clone().slice([0..1,]).squeeze(0)
            .add(self.positional_embedding_matrix.clone().slice([0..seq_len,]))
            .reshape([1, seq_len, embedding_dim]);

        for ix in 1..batch_size {
            let y = inputs.clone().slice([ix..ix+1,]).squeeze(0)
                .add(self.positional_embedding_matrix.clone()
                .slice([ix..ix+1,]))
                .reshape([1, seq_len, embedding_dim]);

                x = Tensor::cat(vec![x, y], 0);
        }
        self.dropout.forward(x)
    }
}