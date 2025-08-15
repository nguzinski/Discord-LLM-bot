# 🤖 Discord LLM Bot

A Discord bot that runs your custom fine-tuned language models with conversation memory, self-chat capabilities, and multiple interaction modes.

## ✨ Features

- **Multiple Chat Interfaces**: Respond to any message beginning with `!`, commands start with `$` try `$helpme`, mentions, and DMs
- **Conversation Memory**: Maintains chat history per channel (last 30 exchanges)
- **Self-Conversation**: Watch two AI personalities chat with each other
- **LoRA Support**: Built for PEFT/LoRA fine-tuned models (base models will also work)
- **Memory Efficient**: 4-bit quantization with BitsAndBytesConfig
- **Thread Safe**: Concurrent request handling with proper locking
- **User Friendly**: Comprehensive help system and error handling
  

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support
- Discord Bot Token
- Fine-tuned language model (preferably with LoRA weights)
- ✨THIS CAN BE DONE WITH JUST A BASE MODEL, GRAB PHI3 or Gemma or Llama and have a blast✨

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/discord-llm-bot.git
   cd discord-llm-bot
   ```

2. **Install dependencies**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install transformers accelerate bitsandbytes peft discord.py
   ```

3. **Configure the bot**
   
   Edit the configuration section at the top of the script:
   
   ```python
   # Model Configuration
   model_name_or_path = "microsoft/DialoGPT-medium"  # Your base model
   lora_weights_path = "./path/to/your/lora/weights"  # Your LoRA weights
   
   # System Prompts
   historyStart = [{"role": "system", "content": "You are a helpful AI assistant named Bob."}]
   
   # Self-Chat Configuration
   author1 = "Alice"
   author2 = "Bob"
   Conv1Start = [{"role": "system", "content": "You are Alice, a curious and energetic AI."}]
   Conv2Start = [{"role": "system", "content": "You are Bob, a thoughtful and analytical AI."}]
   
   # Discord Token
   TOKEN = "YOUR_BOT_TOKEN_HERE"
   ```

4. **Set up Discord Bot**
   - Go to [Discord Developer Portal](https://discord.com/developers/applications)
   - Create a new application and bot
   - Copy the bot token to your configuration
   - Enable "Message Content Intent" in Bot settings
   - Invite bot to your server with appropriate permissions

5. **Run the bot**
   ```bash
   python discord_bot.py
   ```

## 📖 Usage

### Chat Commands

| Command | Description | Example |
|---------|-------------|---------|
| `!<message>` | Main chat interface | `!Hello, how are you?` |
| `@Bot <message>` | Mention the bot | `@Bot what's the weather like?` |
| DM | Direct message the bot | Just send a DM |
| `$chat <message>` | Alternative chat command | `$chat tell me a joke` |

### Bot Management

| Command | Description |
|---------|-------------|
| `$helpme` | Show comprehensive help |
| `$info` | Display bot and model information |
| `$clear` | Clear conversation history for current channel | #may need more testing

### Self-Conversation

| Command | Description |
|---------|-------------|
| `$selfchat` | Start AI vs AI conversation |
| `$stopselfchat` | Stop the self-conversation |
| `$selfchatstatus` | Check if self-chat is running |

## ⚙️ Configuration

### Model Requirements

This bot is designed for:
- **Base Models**: Any HuggingFace compatible causal language model
- **Fine-tuning**: PEFT/LoRA trained models (though base models work too)
- **Memory**:
- 7-8B Recommended 8GB+ VRAM for smooth operation
- 1.7-3B Recommended 3-6GB+ VRAM for smooth operation 

### System Prompts

Configure your bot's personality by editing the system prompts:

```python
# Main conversation personality
historyStart = [{
    "role": "system", 
    "content": "You are a helpful assistant named Claude who loves to chat about technology."
}]

# Self-conversation personalities
Conv1Start = [{
    "role": "system",
    "content": "You are an optimistic AI who always sees the bright side."
}]

Conv2Start = [{
    "role": "system", 
    "content": "You are a pragmatic AI who focuses on practical solutions."
}]
```

### Generation Parameters

Adjust response quality in the `generate_response()` function:

```python
response_ids = model.generate(
    **inputs,
    max_new_tokens=700,        # Response length
    temperature=0.95,          # Creativity (0.1-2.0)
    top_p=0.95,               # Nucleus sampling
    top_k=200,                # Top-k sampling
    no_repeat_ngram_size=3,   # Reduce repetition
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
)
```

## 🔧 Advanced Features

### Memory Management
- **Per-channel memory**: Each Discord channel maintains separate conversation history
- **Automatic trimming**: Keeps last 30 messages to balance context and performance
- **Separate self-chat memory**: Self-conversations use independent memory stores

### Performance Optimization
- **4-bit quantization**: Reduces VRAM usage significantly
- **Thread locking**: Prevents concurrent model access issues
- **Async processing**: Non-blocking Discord interactions
- **CUDA memory management**: Automatic cleanup and optimization

## 🐛 Troubleshooting

### Common Issues

**Bot doesn't respond to messages**
- Ensure "Message Content Intent" is enabled in Discord Developer Portal
- Check bot permissions in your server
- Verify the bot token is correct
- "Try turning it off and on again" 🤓

**CUDA out of memory**
- Reduce `max_new_tokens` in generation config
- Enable quantization (already configured)
- Close other GPU-intensive applications
- Try a smalled Paramater Model 1.7B use only 3-4GB during inference with my setup

**Model loading errors**
- Verify model paths are correct
- Ensure you have sufficient disk space
- Check that your model is compatible with the transformers version

**Self-conversation crashes**
- This usually indicates generation issues
- Try reducing temperature or adjusting other generation parameters
- Check the error logs for specific issues

### Error Messages

```python
# Common error patterns and solutions:

"❌ Error loading LoRA weights"
# Solution: Check lora_weights_path or disable LoRA loading

"oops, had a brain freeze there"
# Solution: Model generation failed, check generation params, (usually tokenizer and parsing problem)

"*self-conversation crashed, whoops*"
# Solution: Usually generation or memory issues, check logs
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the GNU GPL License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

- This bot runs AI models that may generate unpredictable content
- Monitor usage in public servers and implement content filtering as needed
- Ensure compliance with Discord's Terms of Service and Community Guidelines
- Be mindful of computational costs when running on cloud instances

## 🙏 Acknowledgments

- Built with [discord.py](https://github.com/Rapptz/discord.py)
- Powered by [🤗 Transformers](https://github.com/huggingface/transformers)
- Quantization via [BitsAndBytesConfig](https://github.com/TimDettmers/bitsandbytes)
- LoRA support through [PEFT](https://github.com/huggingface/peft)
- Claude for the assist in developement and this README 
---

**Happy chatting! 🚀**
