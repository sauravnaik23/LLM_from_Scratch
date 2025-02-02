'''Contains the configurations for different models'''

GPT2_CONFIG = {
                'vocab_size': 50257,
                'context_len': 1024,
                'embedding_dim': 768,
                'dropout_rate': 0.0,
                'n_layers': 12,
                'n_heads': 12,
                'q_k_v_bias': False,
                'drop_rate': 0.4
              }

CUSTOM_GPT_CONFIG = {'vocab_size': 50257,
 'context_length': 1024,
 'emb_dim': 768,
 'n_heads': 12,
 'n_layers': 12,
 'drop_rate': 0.1,
 'qkv_bias': True}

GPT_355M_CONFIG = {'vocab_size': 50257,
 'context_length': 1024,
 'emb_dim': 1024,
 'n_heads': 16,
 'n_layers': 24,
 'drop_rate': 0.1,
 'qkv_bias': True}
