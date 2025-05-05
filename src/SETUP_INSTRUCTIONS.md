# Setting Up AI Models and Crypto Tools

This document provides step-by-step instructions for setting up and configuring the AI models and cryptocurrency tools integrations.

## Prerequisites

- Python 3.8+ installed
- pip package manager
- Git (for cloning repositories)
- CUDA-compatible GPU (recommended for local models)

## Step 1: Install Dependencies

First, install all the required packages using the provided requirements file:

```bash
pip install -r requirements_ai_crypto.txt
```

This will install all the necessary libraries for both AI models and crypto tools.

## Step 2: Set Up Environment Variables

Create a `.env` file in the root directory with your API keys:

```
# AI Model API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
HF_API_KEY=your_huggingface_api_key_here

# Crypto Trading Bots API Keys
TRALITY_API_KEY=your_trality_api_key_here
THREE_COMMAS_API_KEY=your_3commas_api_key_here
THREE_COMMAS_API_SECRET=your_3commas_api_secret_here
CRYPTOHOPPER_API_KEY=your_cryptohopper_api_key_here

# Crypto Data Providers API Keys
GLASSNODE_API_KEY=your_glassnode_api_key_here
SANTIMENT_API_KEY=your_santiment_api_key_here
CRYPTOPREDICT_API_KEY=your_cryptopredict_api_key_here
AUGMENTO_API_KEY=your_augmento_api_key_here

# Model Paths (for local models)
LLAMA_MODEL_PATH=models/llama-2-7b-chat.gguf
STABLE_DIFFUSION_MODEL_PATH=models/stable-diffusion-v2-1
```

## Step 3: Obtain API Keys

### AI Model API Keys

1. **OpenAI API Key**
   - Go to [platform.openai.com](https://platform.openai.com/)
   - Create an account or sign in
   - Navigate to API keys section
   - Create a new API key and copy it

2. **Anthropic API Key**
   - Go to [console.anthropic.com](https://console.anthropic.com/)
   - Create an account or sign in
   - Request API access
   - Once approved, generate an API key

3. **Hugging Face API Key**
   - Go to [huggingface.co](https://huggingface.co/settings/tokens)
   - Create an account or sign in
   - Generate a new API token

### Crypto Trading API Keys

1. **Trality API Key**
   - Go to [trality.com](https://www.trality.com/)
   - Create an account
   - Navigate to API settings
   - Generate a new API key

2. **3Commas API Key**
   - Go to [3commas.io](https://3commas.io/)
   - Create an account
   - Navigate to API settings
   - Generate both API key and Secret

3. **Cryptohopper API Key**
   - Go to [cryptohopper.com](https://www.cryptohopper.com/)
   - Create an account
   - Navigate to API section
   - Generate a new API key

### Crypto Data API Keys

1. **Glassnode API Key**
   - Go to [glassnode.com](https://glassnode.com/)
   - Create an account
   - Navigate to API section
   - Generate a new API key

2. **Santiment API Key**
   - Go to [app.santiment.net](https://app.santiment.net/)
   - Create an account
   - Go to API section
   - Generate a new API key

## Step 4: Download Local Models (Optional)

For better performance and offline capability, you can download local models:

### LLaMA 2 Model

1. Create a models directory:
   ```bash
   mkdir -p models
   ```

2. Download LLaMA 2 model (requires approval from Meta):
   - Go to [llama.meta.com](https://llama.meta.com/llama-downloads/)
   - Request access
   - Once approved, download the GGUF version of the model
   - Place it in the models directory

3. Alternative method using HuggingFace:
   ```bash
   pip install huggingface_hub
   python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='TheBloke/Llama-2-7B-Chat-GGUF', filename='llama-2-7b-chat.Q4_K_M.gguf', local_dir='models', local_dir_use_symlinks=False)"
   ```

### Stable Diffusion Model

1. Download Stable Diffusion model:
   ```bash
   pip install huggingface_hub
   python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='stabilityai/stable-diffusion-2-1', local_dir='models/stable-diffusion-v2-1')"
   ```

## Step 5: Test the Integrations

Run the example scripts to test your setup:

1. Test AI Models:
   ```bash
   python src/ai_crypto_example.py
   ```

2. Test Image Models:
   ```bash
   python src/image_models_example.py
   ```

3. Test with a specific image:
   ```bash
   python src/image_models_example.py --test-vision --image /path/to/your/image.jpg
   ```

## Troubleshooting

### Common Issues

1. **API key errors**:
   - Ensure API keys are correctly set in the `.env` file
   - Check if the API keys have the necessary permissions
   - Verify the API subscription is active

2. **Package installation errors**:
   - Try installing packages individually to identify problematic dependencies
   - For GPU-related issues, ensure CUDA is properly installed
   - For torch/tensorflow issues, try installing them separately first

3. **Model download issues**:
   - Ensure you have sufficient disk space
   - Try downloading models manually from the respective websites
   - Check network connection and proxy settings

4. **Memory errors with large models**:
   - Try using smaller model variants
   - Reduce batch sizes or image resolutions
   - Use CPU versions if GPU memory is insufficient

### Getting Help

For specific errors or issues, refer to the documentation of the respective libraries:

- OpenAI: [platform.openai.com/docs](https://platform.openai.com/docs)
- Anthropic: [console.anthropic.com/docs](https://console.anthropic.com/docs)
- Hugging Face: [huggingface.co/docs](https://huggingface.co/docs)
- Trading Bot APIs: Check their respective websites for documentation

## Advanced Configuration

### Using Custom Models

To use custom models, modify the respective configuration in your code:

```python
# Custom OpenAI model
gpt_config = {"model": "your-custom-model"}
gpt_model = create_model_interface("openai", gpt_config)

# Custom Hugging Face model
bert_config = {"model_name": "your-custom-bert"}
bert_model = create_model_interface("huggingface", bert_config)
```

### Configuring Crypto Tools

You can customize the crypto tool settings:

```python
# Custom exchange for Trality
trality_config = {"exchange": "binance"}
trality_api = CryptoToolFactory.create_tool("trality", trality_config)
```

## Security Considerations

- Never commit `.env` files or API keys to repositories
- Consider using key rotation for production environments
- Monitor API usage to prevent unexpected charges
- Use read-only API keys when possible for data access

## Updates and Maintenance

- Regularly update the libraries using `pip install -r requirements_ai_crypto.txt --upgrade`
- Check for new model versions on the respective platforms
- Monitor API changes from providers that might require code updates 