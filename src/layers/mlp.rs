use burn::{
    config::Config,
    module::Module,
    nn::{self, Dropout, DropoutConfig, Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};

#[derive(Debug, Module)]
pub struct Mlp<B: Backend> {
    activation_fn: nn::Gelu,
    linear1: Linear<B>,
    linear2: Linear<B>,
    dropout: Dropout,
}

#[derive(Debug, Config)]
pub struct MlpConfig {
    embedding_dim: usize,
    hidden_dim: usize,
    #[config(default = "0.0")]
    dropout: f64,
}

impl MlpConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mlp<B> {
        Mlp {
            activation_fn: nn::Gelu::new(),
            linear1: LinearConfig::new(self.embedding_dim, self.hidden_dim).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            linear2: LinearConfig::new(self.hidden_dim, self.embedding_dim).init(device),
        }
    }
}

impl<B: Backend> Mlp<B> {
    pub fn forward(&self, inputs: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.activation_fn.forward(self.linear1.forward(inputs));
        let x = self.dropout.forward(x);
        let x = self.linear2.forward(x);
        return x;
    }
}
