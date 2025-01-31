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

CUSTOM_GPT_CONFIG = {
                      'vocab_size': 50257,
                      'context_len': 256,
                      'embedding_dim': 78,
                      'dropout_rate': 0.0,
                      'n_layers': 8,
                      'n_heads': 8,
                      'q_k_v_bias': False,
                      'drop_rate': 0.2
                    }