import discord
from discord.ext import commands
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import asyncio
import threading
import random
torch.cuda.empty_cache()
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ====== Load your model and tokenizer ======

model_name_or_path = "THE HUGGING FACE ID OF THE BASE MODEL"
lora_weights_path = "WHERE YOUR PATH TO WEIGHTS IS"

#######################
#
# Per-channel conversation memory - Regular conversations
#
#"role": "system", "content": "Your system prompt you want saved into memory of the model"
#
#EXAMPLE: "role": "system", "content": "You are a Nick, always act like Nick, not an AI or assistant"  #if your training data was parsed for someone named Nick for example
#
# 

historyStart = [{"PUT ABOVE HERE"}]

# Self-conversation memory stores
#This is a system for two seperate memory stores, to allow a bot to talk and converse with itself
#
#These two prompts can be the same or different, its all up to your purposes and your training data
#set these to the names of the bots you want to talk to each other, if you plan on using '$selfchat'
author1 = "bot1"
author2 = "bot2"

Conv1Start = [{
    "PUT ABOVE HERE"
}]

Conv2Start = [{
    "PUT ABOVE HERE or Put another Prompt"
}]
#######################


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# Add pad token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model with quantization and device mapping...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="auto",
    #I found that swithcing to bnb config gave me a little less VRAM usage
    # Could be just instability in a few runs
    quantization_config=bnb_config,
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

# My Model was trained with Peft QLora, yours may not be, its up to how you trained it
# this wrapper can be turned off if yours is not a Peft Model, or left to load through the exception
print("Loading PEFT LoRA weights...")
try:
    model = PeftModel.from_pretrained(base_model, lora_weights_path)
    print("✅ LoRA weights loaded successfully!")
except Exception as e:
    print(f"❌ Error loading LoRA weights: {e}")
    print("Using base model without LoRA...")
    model = base_model

model.eval()
print("Model ready for Discord!")

# ====== Discord bot setup ======
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="$", intents=intents)  # Changed prefix to $ to avoid conflicts

# Self-conversation control
self_conversations = {}  # {channel_id: {'active': bool, 'task': asyncio.Task, 'turn': int}}

# Thread lock for model inference
model_lock = threading.Lock()

def generate_response(author, user_input, channel_id, memory_store=None):
    """Generate response using the model (runs in separate thread)"""
    global historyStart
    
    # Use specified memory store or default
    # Could be used for loading in a textfile, a more advanced and permenant memory than I have implemented here

    current_history = memory_store if memory_store is not None else historyStart
    
    # Build messages using role system instead of string concatenation
    messages = []
    
    # Add conversation history (keep recent messages only)
    # Only add history if it exists and limit to recent messages
    # Too much memory and the model doesnt understand whats important too well,
    # Too little and the model doesnt remember 15 seconds ago
    if current_history:
        recent_history = current_history[-30:]  # Keep last 30 messages
        messages = recent_history + messages
    
    # Add current user input
    # useful if you have training data from different users
    messages.append({
        "role": "user", 
        "content": f"{author}: {user_input}"  # Include author name for context
    })
    
    # Apply chat template this makes the whole "role": "content": thing, work for whatever model you happen to throw in here
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    
    # Generate response with thread lock
    with model_lock:
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        response_ids = model.generate(
            **inputs,
            max_new_tokens=700,
            no_repeat_ngram_size=3,
            top_p=0.95,
            temperature=0.95,
            top_k=200,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )[0][len(inputs.input_ids[0]):].tolist()
        
        response = tokenizer.decode(response_ids, skip_special_tokens=True).strip() # here you can filter for unwanted output manually if you wish, .replace() profanity if you are having trouble tuning it out
        while response == None:
            response = get_model_response(author, user_input, channel_id, memory_store)
    
    # Update the appropriate history
    target_history = memory_store if memory_store is not None else historyStart
    
    # Add user message to history
    target_history.append({
        "role": "user",
        "content": f"{author}: {user_input}"
    })
    
    # Add assistant response to history  
    target_history.append({
        "role": "assistant",
        "content": response
    })
    
    # Trim history to prevent it from getting too long
    max_history_length = 30
    if len(target_history) > max_history_length:
        target_history = target_history[-max_history_length:]
    
    return response

async def get_model_response(author, user_input, channel_id, memory_store=None):
    """Async wrapper for model generation"""
    try:
        # Run model generation in a separate thread to avoid blocking
        response = await asyncio.to_thread(generate_response, author, user_input, channel_id, memory_store)
        return response
    except Exception as e:
        print(f"Error in async response generation: {e}")
        #If you just crtl+f'd to find whats causing this error, it means generation is messed up, can be a tokenizer issue or how you are parsing, goodluck
        return "oops, had a brain freeze there. what were we talking about?"

async def self_conversation_loop(channel):
    """Main loop for self-conversation"""
    channel_id = channel.id
    turn = 0  # 0 = bot1's turn, 1 = bot2's turn
    
    # Conversation starters
    # bot1 will start with this text and then bot2 will respond to it, conversations diverge pretty quickly if temperature is high enough
    # makes for good reality TV
    # you can add multiple
    starters = [
        "Hey what are you doing?",
        "Guess what I just did!",
    ]
    
    # Start with a random conversation starter
    current_message = random.choice(starters)
    
    try:
        while self_conversations[channel_id]['active']:
            if turn == 0:
                # Bot1's turn
                author = author1
                memory = self_conv_history_1
                
                # Send message as bot
                await channel.send(f"**{author}:** {current_message}")
                
                # Generate bot1's response 
                next_message = await get_model_response(author2, current_message, channel_id, Conv1Start)
                current_message = next_message
                turn = 1
                
            else:
                # bot2's turn  
                author = author2
                memory = self_conv_history_2
                
                # Send message as bot
                await channel.send(f"**{author}:** {current_message}")
                
                # Generate bot2's response
                next_message = await get_model_response(author1, current_message, channel_id, Conv2Start)
                current_message = next_message
                turn = 0
            
            self_conversations[channel_id]['turn'] = turn
            
            # Wait 4 seconds before next response
            await asyncio.sleep(4)
            
    except asyncio.CancelledError:
        # Conversation was stopped
        pass
    except Exception as e:
        print(f"Error in self-conversation: {e}")
        await channel.send("*self-conversation crashed, whoops*")
    finally:
        # Clean up
        if channel_id in self_conversations:
            self_conversations[channel_id]['active'] = False

# ====== Event handlers ======
@bot.event
async def on_ready():
    print(f"✅ {bot.user} is now online and ready!")
    print(f"Bot is in {len(bot.guilds)} servers")
    
    # Set bot status
    activity = discord.Activity(type=discord.ActivityType.listening, name="!message or $helpme")
    await bot.change_presence(activity=activity)

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    
    if message.content.startswith("!"):
        user_input = message.content[1:].strip()  
        if not user_input:
            await message.channel.send("what do you want? say something after the '!'")
            return
        
        async with message.channel.typing():
            response = await get_model_response(message.author, user_input, message.channel.id)
            await message.channel.send(response)
        return
    

    await bot.process_commands(message)
    
    # Handle direct messages or mentions without command prefix
    if isinstance(message.channel, discord.DMChannel) or bot.user in message.mentions:
        user_input = message.content.replace(f"<@{bot.user.id}>", "").strip()
        if user_input:
            async with message.channel.typing():
                response = await get_model_response(user_input, message.channel.id)
                await message.channel.send(response)

@bot.command(name="selfchat")
async def start_self_conversation(ctx):
    """Start a self-conversation between Bot and Bot2"""
    channel_id = ctx.channel.id
    
    # Check if already running
    if channel_id in self_conversations and self_conversations[channel_id]['active']:
        await ctx.send("We're already talking to ourselves here! use `$stopselfchat` to stop it first")
        return
    
    # Start the conversation
    self_conversations[channel_id] = {
        'active': True,
        'turn': 0,
        'task': None
    }
    
    # Clear the self-conversation histories for a fresh start

    global self_conv_history_1, self_conv_history_2
    self_conv_history_1 = Conv1Start
    
    self_conv_history_2 = Conv2Start
    
    await ctx.send(f"🗣️ **Starting self-conversation!** {author1} vs {author2}, let's see what happens...")
    
    # Start the conversation loop
    task = asyncio.create_task(self_conversation_loop(ctx.channel))
    self_conversations[channel_id]['task'] = task

@bot.command(name="stopselfchat")
async def stop_self_conversation(ctx):
    """Stop the self-conversation"""
    channel_id = ctx.channel.id
    
    if channel_id not in self_conversations or not self_conversations[channel_id]['active']:
        await ctx.send("no self-chat running here")
        return
    
    # Stop the conversation
    self_conversations[channel_id]['active'] = False
    
    # Cancel the task if it exists
    if self_conversations[channel_id]['task']:
        self_conversations[channel_id]['task'].cancel()
    
    # Clean up
    del self_conversations[channel_id]
    
    await ctx.send("🛑 **Self-conversation stopped!** Back to normal chatting")

@bot.command(name="selfchatstatus")
async def self_chat_status(ctx):
    """Check if self-conversation is running"""
    channel_id = ctx.channel.id
    
    if channel_id in self_conversations and self_conversations[channel_id]['active']:
        turn_name = author1 if self_conversations[channel_id]['turn'] == 0 else author2
        await ctx.send(f"🗣️ Self-chat is **active**! Next up: {turn_name}")
    else:
        await ctx.send("No self-chat running. Use `$selfchat` to start one!")

@bot.command(name="chat")
async def chat_command(ctx, *, message=None):
    """Chat with Bot! Usage: $chat your message here"""
    if not message:
        await ctx.send("what do you want to talk about? use `$chat your message here`")
        return
    
    async with ctx.typing():
        response = await get_model_response(message, ctx.channel.id)
        await ctx.send(response)

@bot.command(name="clear")
async def clear_history(ctx):
    """Clear conversation history for this channel"""

    global history, self_conv_history_1, self_conv_history_2
    
        # Reset main history
    # Per-channel conversation memory - Regular conversations
    #
    #"role": "system", "content": "Your system prompt you want saved into memory of the model"
    #
    #EXAMPLE: "role": "system", "content": "You are a Nick, always act like Nick, not an AI or assistant"  #if your training data was parsed for someone named Nick for example
    #
    #


    # Self-conversation memory stores
    #
    ##EXAMPLE: "role": "system", "content": "You are a Nick, always act like Nick, not an AI or assistant"
    #
    #This is a system for two seperate memory stores, to allow a bot to talk and converse with itself
    #
    #These two prompts can be the same or different, its all up to your purposes and your training data

    self_conv_history_1 = [{
        "PUT ABOVE HERE"
    }]

    self_conv_history_2 = [{
        "PUT ABOVE HERE or Put another Prompt"
    }]

        
    await ctx.send("cleared all my memories for this channel! fresh start 🧠✨")

@bot.command(name="info")
async def bot_info(ctx):
    """Show bot information"""
    embed = discord.Embed(
        title="🤖 Bot Info",
        color=discord.Color.blue()
    )
    embed.add_field(
        name="Model", 
        value=f"{model_name_or_path}\n+ LoRA fine-tuning", 
        inline=False
    )
    embed.add_field(
        name="Commands",
        value="• `!your message` - Main chat (just type ! then your message)\n• `$selfchat` - Start self-conversation\n• `$stopselfchat` - Stop self-conversation\n• `$clear` - Clear conversation history\n• `$info` - Show this info\n• Or mention me or DM me!",
        inline=False
    )
    embed.add_field(
        name="Memory",
        value=f"Remembers last 30 exchanges per channel\nSeparate memories for self-chat",
        inline=False
    )
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        vram_used = torch.cuda.memory_allocated() / 1024**3
        embed.add_field(
            name="Hardware",
            value=f"GPU: {gpu_name}\nVRAM: {vram_used:.1f}GB",
            inline=False
        )
    
    await ctx.send(embed=embed)

@bot.command(name="helpme")
async def help_command(ctx):
    """Show help message"""
    help_text = """
🤖 **Bot - How to Chat**

**Main way to talk to me:**
• `!your message here` - Just type ! followed by anything you want to say

**Self-Conversation Commands:**
• `$selfchat` - Start a conversation between the two system prompts you provided
• `$stopselfchat` - Stop the self-conversation
• `$selfchatstatus` - Check if self-chat is running

**Other ways to chat:**
• Mention me: `@[Bots name] hey what's up`  
• Send me a DM - I'll respond automatically

**Bot commands (use $ prefix):**
• `$clear` - Clear conversation history for this channel
• `$info` - Technical info about the bot
• `$helpme` - Show this message

**Tips:**
• I remember our conversation history (last 30 exchanges per channel)
• Self-conversations use separate memory stores for each personality
• I work best with casual, friendly messages
• If I'm being weird, try `$clear` to reset my memory
    """
    await ctx.send(help_text)

# ====== Error handling ======
@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        # Ignore unknown commands
        return
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send("looks like you forgot something. try `$helpme` to see how to use commands")
    else:
        print(f"Command error: {error}")
        await ctx.send("something went wrong. try again or use `$helpme`")

# ====== Run the bot ======
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 Starting Discord Bot")
    print("="*50)
    print("Make sure to:")
    print("1. Set your bot token below")
    print("2. Invite the bot to your server with message permissions")
    print("3. Enable message content intent in Discord Developer Portal")
    print("="*50)
    
  
    TOKEN = "YOUR_BOT_TOKEN_HERE"  # Remember to replace this with your actual token
    
    if historyStart == [{"PUT ABOVE HERE"}]:
        print("❌ Please set conversation system prompt, at the very top")
        exit
    elif TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("❌ Please set your Discord bot token!")
        print("Get one from: https://discord.com/developers/applications")
    else:
        try:
            bot.run(TOKEN)
        except discord.LoginFailure:
            print("❌ Invalid bot token!")
        except Exception as e:
            print(f"❌ Error starting bot: {e}")