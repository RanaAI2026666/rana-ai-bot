import os
import io
import base64
import asyncio
import tempfile
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
IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-2.5-flare")
TTS_MODEL = os.environ.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = os.environ.get("OPENAI_TTS_VOICE", "marin")

intents = discord.Intents.default()
intents.message_content = True
bot = discord.Client(intents=intents)


def extract_text(data):
    return "".join(
        item.get("text", "")
        for output in data.get("output", [])
        for item in output.get("content", [])
        if item.get("type") == "output_text"
    ).strip()


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
    return extract_text(data)


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
        return extract_text(data)


async def generate_image(session, prompt):
    payload = {
        "model": IMAGE_MODEL,
        "prompt": prompt[:32000],
        "size": "1024x1024",
        "quality": "medium",
        "n": 1,
        "output_format": "png"
    }
    async with session.post(
        "https://api.openai.com/v1/images/generations",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=aiohttp.ClientTimeout(total=240)
    ) as response:
        data = await response.json()
        if response.status >= 400:
            raise RuntimeError(data.get("error", {}).get("message", "فشل إنشاء الصورة"))
    encoded = data.get("data", [{}])[0].get("b64_json")
    if not encoded:
        raise RuntimeError("لم تصل الصورة من خدمة الإنشاء.")
    return base64.b64decode(encoded)


async def generate_speech(session, text):
    if len(text) > 3500:
        text = text[:3500]
    payload = {
        "model": TTS_MODEL,
        "input": text,
        "voice": TTS_VOICE,
        "instructions": "Speak Arabic naturally and warmly, with a friendly conversational tone.",
        "response_format": "mp3"
    }
    async with session.post(
        "https://api.openai.com/v1/audio/speech",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=aiohttp.ClientTimeout(total=240)
    ) as response:
        if response.status >= 400:
            try:
                data = await response.json()
                detail = data.get("error", {}).get("message", "فشل توليد الصوت")
            except Exception:
                detail = "فشل توليد الصوت"
            raise RuntimeError(detail)
        return await response.read()


def command_prompt(content, prefixes):
    lower = content.lower().strip()
    for prefix in prefixes:
        if lower.startswith(prefix):
            return content[len(prefix):].strip()
    return None


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

    content = (message.content or "").strip()
    if not content and not message.attachments:
        return

    if not OPENAI_API_KEY and any(x in content.lower() for x in ("!صورة", "!صوت", "!image", "!voice")):
        await message.reply("ميزة الإنشاء تحتاج ضبط مفتاح OpenAI في إعدادات البوت.")
        return

    image_prompt = command_prompt(content, ("!صورة", "!image", "ارسم صورة:", "أنشئ صورة:", "انشئ صورة:"))
    speech_text = command_prompt(content, ("!صوت", "!voice", "اقرأ بصوت:", "حوّل إلى صوت:", "حول الى صوت:"))
    video_prompt = command_prompt(content, ("!فيديو", "!video", "أنشئ فيديو:", "انشئ فيديو:"))

    try:
        async with aiohttp.ClientSession() as session:
            if image_prompt is not None:
                if not image_prompt:
                    await message.reply("اكتب وصف الصورة بعد الأمر، مثل: !صورة قطة صغيرة في حديقة.")
                    return
                async with message.channel.typing():
                    image_bytes = await generate_image(session, image_prompt)
                await message.reply(
                    content="هذه الصورة التي أنشأتها لك. الصورة مولّدة بالذكاء الاصطناعي.",
                    file=discord.File(io.BytesIO(image_bytes), filename="rana-image.png")
                )
                return

            if speech_text is not None:
                if not speech_text:
                    await message.reply("اكتب النص بعد الأمر، مثل: !صوت مرحبًا، كيف حالك؟")
                    return
                async with message.channel.typing():
                    audio_bytes = await generate_speech(session, speech_text)
                await message.reply(
                    content="هذا تسجيل صوتي مولّد بالذكاء الاصطناعي بصوت رنا.",
                    file=discord.File(io.BytesIO(audio_bytes), filename="rana-voice.mp3")
                )
                return

            if video_prompt is not None:
                await message.reply(
                    "أقدر حاليًا أنشئ لك صورة وأرسلها، وأحوّل النص إلى صوت. "
                    "لكن خدمة إنشاء الفيديو عبر OpenAI API أُوقفت، لذلك لا أستطيع إنشاء MP4 حقيقي الآن."
                )
                return

            attachment_context = await process_attachments(message, session)
            user_text = content
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
        await message.reply("عذرًا، واجهت مشكلة أثناء معالجة طلبك. حاول مرة أخرى بعد قليل.")


bot.run(DISCORD_TOKEN)
