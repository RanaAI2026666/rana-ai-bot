
import os
import discord
import aiohttp

DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
NOMI_API_KEY = os.environ["NOMI_API_KEY"]
NOMI_ID = os.environ["NOMI_ID"]

intents = discord.Intents.default()
intents.message_content = True

bot = discord.Client(intents=intents)

@bot.event
async def on_ready():
    print(f"Rana AI is online: {bot.user}")

@bot.event
async def on_message(message):
    if message.author.bot:
        return

    headers = {
        "Authorization": NOMI_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {"messageText": message.content}

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"https://api.nomi.ai/v1/nomis/{NOMI_ID}/chat",
            headers=headers,
            json=payload
        ) as response:
            data = await response.json()

    reply = data.get("replyMessage", {}).get("text")
    if reply:
        await message.reply(reply[:2000])

bot.run(DISCORD_TOKEN)
