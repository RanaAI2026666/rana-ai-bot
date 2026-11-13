import os
import io
import base64
import asyncio
import tempfile
import subprocess
from pathlib import Path

import discord
import aiohttp
import imageio_ffmpeg

DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
NOMI_API_KEY = os.environ["NOMI_API_KEY"]
NOMI_ID = os.environ["NOMI_ID"]
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024
OPENAI_MODEL = "gpt-4.1-mini"

intents = discord.Intents.default()
intents.message_content = True
bot = discord.Client(intents=intents)


async def openai_image_analysis(session, image_url, prompt):
    payload = {
        "model": OPENAI_MODEL,
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt or "صف هذه الصورة وأجب عن سؤال المستخدم بالعربية."},
                {"type": "input_image", "image_url": image_url, "detail": "auto"}
            ]
        }]
    }
    async with session.post(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=aiohttp.ClientTimeout(total=90)
    ) as response:
        data = await response.json()
        if response.status >= 400:
            raise RuntimeError(data.get("error", {}).get("message", "OpenAI image request failed"))
    return "".join(
        item.get("text", "")
        for output in data.get("output", [])
        for item in output.get("content", [])
        if item.get("type") == "output_text"
    ).strip()


async def openai_transcribe(session, attachment):
    raw = await attachment.read()
    if len(raw) > MAX_ATTACHMENT_BYTES:
        raise ValueError("حجم التسجيل أكبر من الحد المسموح (20 ميجابايت).")
    form = aiohttp.FormData()
    form.add_field("model", "gpt-transcribe")
    form.add_field("file", raw, filename=attachment.filename or "audio.ogg",
                   content_type=attachment.content_type or "application/octet-stream")
    async with session.post(
        "https://api.openai.com/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        data=form,
        timeout=aiohttp.ClientTimeout(total=180)
    ) as response:
        data = await response.json()
        if response.status >= 400:
            raise RuntimeError(data.get("error", {}).get("message", "OpenAI transcription failed"))
    return data.get("text", "").strip()


async def openai_video_analysis(session, attachment, prompt):
    raw = await attachment.read()
    if len(raw) > MAX_ATTACHMENT_BYTES:
        raise ValueError("حجم الفيديو أكبر من الحد المسموح (20 ميجابايت).")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / "clip.mp4"
        video_path.write_bytes(raw)
        # Sample four representative frames across the clip.
        cmd = [ffmpeg, "-v", "error", "-i", str(video_path), "-vf",
               "fps=1/3,scale=768:-1", "-frames:v", "4", str(Path(tmp) / "frame-%02d.jpg")]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError("تعذر استخراج صور من الفيديو: " + stderr.decode(errors="ignore")[:300])
        frames = sorted(Path(tmp).glob("frame-*.jpg"))
        if not frames:
            raise ValueError("لم أستطع قراءة إطارات الفيديو.")
        content = [{"type": "input_text", "text":
                    (prompt + "\n" if prompt else "") +
                    "حلّل إطارات هذا الفيديو بالترتيب، واشرح ما يحدث بالعربية. هذه عينات من الفيديو وليست كل لحظاته."}]
        for frame in frames:
            encoded = base64.b64encode(frame.read_bytes()).decode("ascii")
            content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{encoded}", "detail": "auto"})
        payload = {"model": OPENAI_MODEL, "input": [{"role": "user", "content": content}]}
        async with session.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json=payload,
            timeout=aiohttp.ClientTimeout(total=180)
        ) as response:
            data = await response.json()
            if response.status >= 400:
                raise RuntimeError(data.get("error", {}).get("message", "OpenAI video request failed"))
        return "".join(
            item.get("text", "")
            for output in data.get("output", [])
            for item in output.get("content", [])
            if item.get("type") == "output_text"
        ).strip()


async def process_attachments(message, session):
    notes = []
    if not message.attachments:
        return ""
    if not OPENAI_API_KEY:
        return "ملاحظة: ميزات الصور والصوت والفيديو غير مفعّلة لأن مفتاح OpenAI غير مضبوط."

    for attachment in message.attachments[:3]:
        ctype = (attachment.content_type or "").lower()
        name = (attachment.filename or "").lower()
        try:
            if ctype.startswith("image/") or name.endswith((".png", ".jpg", ".jpeg", ".webp")):
                if attachment.size > MAX_ATTACHMENT_BYTES:
                    notes.append(f"{name}: الملف أكبر من 20 ميجابايت.")
                    continue
                result = await openai_image_analysis(
                    session, attachment.url,
                    message.content or "اشرح الصورة وأجب بالعربية."
                )
                notes.append("تحليل الصورة: " + result)
            elif ctype.startswith("audio/") or name.endswith((".mp3", ".wav", ".m4a", ".ogg", ".webm", ".flac")):
                transcript = await openai_transcribe(session, attachment)
                notes.append("تفريغ التسجيل الصوتي: " + (transcript or "لم يتم التعرف على كلام واضح."))
            elif ctype.startswith("video/") or name.endswith((".mp4", ".mov", ".m4v", ".webm")):
                result = await openai_video_analysis(session, attachment, message.content)
                notes.append("تحليل الفيديو: " + result)
            else:
                notes.append(f"المرفق {name or 'غير معروف'}: نوع الملف غير مدعوم حاليًا.")
        except Exception as exc:
            notes.append(f"تعذر تحليل {name or 'المرفق'}: {str(exc)[:350]}")
    return "\n".join(notes)


@bot.event
async def on_ready():
    print(f"Rana AI is online: {bot.user}")


@bot.event
async def on_message(message):
    if message.author.bot:
        return

    try:
        async with aiohttp.ClientSession() as session:
            attachment_context = await process_attachments(message, session)
            user_text = message.content or ""
            if attachment_context:
                user_text = (user_text + "\n\n" + attachment_context).strip()
            if not user_text:
                return

            headers = {
                "Authorization": NOMI_API_KEY,
                "Content-Type": "application/json"
            }
            payload = {"messageText": user_text}
            async with session.post(
                f"https://api.nomi.ai/v1/nomis/{NOMI_ID}/chat",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=90)
            ) as response:
                data = await response.json()
                if response.status >= 400:
                    raise RuntimeError(data.get("message", "Nomi request failed"))

        reply = data.get("replyMessage", {}).get("text")
        if reply:
            await message.reply(reply[:2000])
        elif attachment_context:
            await message.reply(attachment_context[:2000])
    except Exception as exc:
        print(f"Message processing error: {type(exc).__name__}: {exc}")
        await message.reply("عذرًا، واجهت مشكلة أثناء معالجة رسالتك. حاول مرة أخرى بعد قليل.")


bot.run(DISCORD_TOKEN)
