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
