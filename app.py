from flask import Flask, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi
import re
from openai import AsyncOpenAI, OpenAIError
import os
from auth import require_custom_authentication
from dotenv import load_dotenv
import logging
import asyncio
import tiktoken

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up OpenAI client
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_youtube_id(url):
    video_id = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    return video_id.group(1) if video_id else None

def process_transcript(video_id):
    proxy_address = os.environ.get("PROXY")
    transcript = YouTubeTranscriptApi.get_transcript(
        video_id,
        proxies={"http": proxy_address, "https": proxy_address}
    )
    return ' '.join([entry['text'] for entry in transcript])

def chunk_text(text, max_tokens=16000):
    tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
    words = text.split()
    chunks = []
    current_chunk = []
    current_token_count = 0

    for word in words:
        word_token_count = len(tokenizer.encode(word + " "))
        if current_token_count + word_token_count > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_token_count = 0
        current_chunk.append(word)
        current_token_count += word_token_count

    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

async def process_chunk(chunk):
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """You are a helpful assistant that improves text formatting and adds punctuation. 
                 You will be given texts from YouTube transcriptions and your task is to apply good formatting.
                 Do NOT modify individual words."""},
                {"role": "user", "content": chunk}
            ]
        )
        return response.choices[0].message.content
    except OpenAIError as e:
        return f"OpenAI API error: {str(e)}"

async def improve_text_with_gpt4(text):
    if not client.api_key:
        return "OpenAI API key not found."
    chunks = chunk_text(text)
    tasks = [process_chunk(chunk) for chunk in chunks]
    improved_chunks = await asyncio.gather(*tasks)
    return ' '.join(improved_chunks)

# 🔐 Authenticated POST endpoint
@app.route('/transcribe', methods=['POST'])
@require_custom_authentication
def transcribe():
    youtube_url = request.json.get('url')
    if not youtube_url:
        return jsonify({"error": "No YouTube URL provided"}), 400

    video_id = get_youtube_id(youtube_url)
    if not video_id:
        return jsonify({"error": "Invalid YouTube URL"}), 400

    try:
        logger.info(f"videoid = {video_id}")
        transcript_text = process_transcript(video_id)
        improved_text = asyncio.run(improve_text_with_gpt4(transcript_text))
        return jsonify({"result": improved_text})
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return jsonify({"error": "Internal server error"}), 500

# 🌐 Public GET endpoint
@app.route('/transcript/<video_id>', methods=['GET'])
def get_transcript(video_id):
    try:
        logger.info(f"videoid = {video_id}")
        transcript_text = process_transcript(video_id)
        improved_text = asyncio.run(improve_text_with_gpt4(transcript_text))
        return jsonify({"result": improved_text})
    except Exception as e:
        logger.exception(f"Error processing transcript: {e}")
        return jsonify({"error": "Failed to fetch or process transcript."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
