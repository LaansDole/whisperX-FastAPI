# Language-Optimized Transcription API

This API provides automatic model selection based on AudioBench performance data for Southeast Asian languages.

## New Endpoints

### `/speech-to-text-optimized`
Automatically selects the optimal model for your specified language.

**Parameters:**
- `language` (required): Language code (e.g., 'vi', 'en', 'zh', 'yue')
- `task` (optional): 'transcribe' or 'translate' (default: 'transcribe')
- `device` (optional): 'cuda' or 'cpu' (default: 'cuda')
- `override_model` (optional): Force a specific model instead of auto-selection
- All existing alignment, diarization, and ASR parameters

### `/speech-to-text-url-optimized`
Same as above but processes audio from a URL.

## Language Model Selection

Based on AudioBench leaderboard results, the system automatically selects:

### 🇸🇪 **Southeast Asian Languages** → **MERaLiON SEA-LION**
The MERaLiON/MERaLiON-AudioLLM-Whisper-SEA-LION model is the **AudioBench best performer** for:
- **Vietnamese** (`vi`, `vie`)
- **English** (`en`, `eng`) - especially Singapore English
- **Mandarin Chinese** (`zh`, `cmn`) 
- **Cantonese** (`yue`)
- **Singapore English** (`en-sg`, `sg-en`)
- **Singlish** - colloquial Singapore English

**Why MERaLiON SEA-LION?**
- ✅ **AudioBench #1** for Southeast Asian languages
- ✅ Trained on **260,000 hours** of SEA speech data
- ✅ Fine-tuned for **local accents and dialects**
- ✅ Superior performance on **IMDA Singapore Speech Corpus**
- ✅ Optimized for **multilingual code-switching** common in SEA

### 🌍 **All Other Languages** → **Whisper large-v3-turbo**
- Uses OpenAI's best general-purpose Whisper model
- Fast inference with excellent accuracy
- Supports 100+ languages

## Example Usage

### Southeast Asian Languages (using MERaLiON SEA-LION)

```bash
# Vietnamese - automatically uses MERaLiON SEA-LION
curl -X POST "http://localhost:8000/speech-to-text-optimized" \
  -F "file=@vietnamese_audio.wav" \
  -F "language=vi" \
  -F "enable_automated_diarization=true"

# Singapore English - automatically uses MERaLiON SEA-LION  
curl -X POST "http://localhost:8000/speech-to-text-optimized" \
  -F "file=@singlish_audio.wav" \
  -F "language=en-sg" \
  -F "task=transcribe"

# Mandarin Chinese - automatically uses MERaLiON SEA-LION
curl -X POST "http://localhost:8000/speech-to-text-optimized" \
  -F "file=@mandarin_audio.wav" \
  -F "language=zh" \
  -F "task=transcribe"

# Cantonese - automatically uses MERaLiON SEA-LION
curl -X POST "http://localhost:8000/speech-to-text-optimized" \
  -F "file=@cantonese_audio.wav" \
  -F "language=yue" \
  -F "task=transcribe"
```

### Other Languages (using Whisper large-v3-turbo)

```bash
# Spanish - automatically uses Whisper large-v3-turbo
curl -X POST "http://localhost:8000/speech-to-text-optimized" \
  -F "file=@spanish_audio.wav" \
  -F "language=es" \
  -F "enable_automated_diarization=true"

# Arabic - automatically uses Whisper large-v3-turbo
curl -X POST "http://localhost:8000/speech-to-text-optimized" \
  -F "file=@arabic_audio.wav" \
  -F "language=ar" \
  -F "task=transcribe"

# Override auto-selection (force specific model)
curl -X POST "http://localhost:8000/speech-to-text-optimized" \
  -F "file=@audio.wav" \
  -F "language=vi" \
  -F "override_model=large-v3-turbo"
```

## Performance Benefits

### 🏆 **MERaLiON SEA-LION Advantages:**
- **Superior Accuracy**: AudioBench-proven best for SEA languages
- **Cultural Nuances**: Handles local accents, code-switching, and colloquialisms
- **Specialized Training**: 260k hours of SEA speech data
- **Multilingual Support**: Seamlessly handles language mixing

### ⚡ **System Optimizations:**
- **Automatic Selection**: No manual model research needed
- **Compute Optimization**: `float16` for SEA languages, `int8` for CPU
- **Diarization Tuning**: SEA conversations often have more speakers (12-15 vs 8)
- **Confidence Adjustment**: Lower thresholds for challenging SEA accents

## Response Format

The optimized endpoints return additional metadata about model selection:

```json
{
  "segments": [...],
  "language": "vi",
  "optimization_metadata": {
    "selected_model": "MERaLiON/MERaLiON-AudioLLM-Whisper-SEA-LION",
    "language": "vi", 
    "compute_type": "float16",
    "optimization_applied": true,
    "device": "cuda"
  },
  "optimization_info": {
    "workflow_type": "optimized",
    "language": "vi",
    "model_override": null,
    "optimization_enabled": true
  }
}
```

## Supported Languages

### MERaLiON SEA-LION (AudioBench Best)
| Language | Code | Notes |
|----------|------|-------|
| Vietnamese | `vi`, `vie` | AudioBench #1 |
| English | `en`, `eng` | Singapore English optimized |
| Mandarin | `zh`, `cmn` | Standard Mandarin |
| Cantonese | `yue` | Traditional Cantonese |
| Singapore English | `en-sg`, `sg-en` | Singlish support |

### Whisper large-v3-turbo (Default)
- All other languages supported by Whisper (100+ languages)
- Best general-purpose performance
- Fast inference speed

## Technical Details

### MERaLiON SEA-LION Architecture:
- **Audio Encoder**: MERaLiON-Whisper (fine-tuned from Whisper-large-v2)
- **Text Decoder**: SEA-LION V3 (AI Singapore's localized LLM)
- **Training**: 260,000 hours of multilingual SEA speech data
- **Tasks**: ASR, Speech Translation, Q&A, Dialogue Summarization, Instruction Following, Paralinguistics

### AudioBench Performance:
MERaLiON SEA-LION significantly outperforms competitors on:
- **MNSC-ASR**: Singapore Multitask National Speech Corpus
- **Common Voice**: Southeast Asian language datasets  
- **Speech Translation**: SEA language pairs
- **Code-switching**: Typical SEA multilingual conversations

## Migration from Standard Endpoints

To migrate from standard endpoints:
1. Replace `/speech-to-text` with `/speech-to-text-optimized`
2. Add required `language` parameter
3. Remove optional `model` parameter (unless using `override_model`)
4. All other parameters remain the same

**Example:**
```bash
# Before
curl -X POST "/speech-to-text" -F "language=vi" -F "model=large-v3"

# After (automatically uses MERaLion SEA-LION for Vietnamese)
curl -X POST "/speech-to-text-optimized" -F "language=vi"
```

The optimized endpoints automatically provide the **AudioBench-proven best model** for Southeast Asian languages and excellent general-purpose performance for all others.