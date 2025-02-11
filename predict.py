from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
from PIL import Image
from deepseek_vl2.serve.inference import (
    convert_conversation_to_prompts,
    deepseek_generate,
    load_model,
)
from deepseek_vl2.models.conversation import SeparatorStyle

MODEL_CACHE = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/deepseek-ai/deepseek-vl2-small/model.tar"
# MODEL_URL = "https://weights.replicate.delivery/default/deepseek-ai/deepseek-vl2/model.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Download the weights if they don't exist
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        print("Loading model...")
        self.tokenizer, self.vl_gpt, self.vl_chat_processor = load_model(MODEL_CACHE, dtype=torch.bfloat16)
        print("Model loaded successfully")

    def predict(
        self,
        image: Path = Input(description="Input image file"),
        prompt: str = Input(description="Text prompt to guide the model"),
        temperature: float = Input(
            description="Temperature for sampling", default=0.1, ge=0.0, le=1.0
        ),
        top_p: float = Input(
            description="Top-p sampling parameter", default=0.9, ge=0.0, le=1.0
        ),
        repetition_penalty: float = Input(
            description="Repetition penalty", default=1.1, ge=0.0, le=2.0
        ),
        max_length_tokens: int = Input(
            description="Maximum number of tokens to generate",
            default=2048,
            ge=0,
            le=4096,
        ),
    ) -> str:
        """Run a single prediction on the model"""
        try:
            # Load and convert image
            pil_image = Image.open(str(image)).convert("RGB")
            
            # Initialize conversation
            conversation = self.vl_chat_processor.new_chat_template()
            
            # Add image and prompt to conversation
            conversation.append_message(conversation.roles[0], (prompt, [pil_image]))
            conversation.append_message(conversation.roles[1], "")

            # Convert conversation to model inputs
            all_conv, last_image = convert_conversation_to_prompts(conversation)
            stop_words = conversation.stop_str

            # Generate response
            full_response = ""
            with torch.no_grad():
                for x in deepseek_generate(
                    conversations=all_conv,
                    vl_gpt=self.vl_gpt,
                    vl_chat_processor=self.vl_chat_processor,
                    tokenizer=self.tokenizer,
                    stop_words=stop_words,
                    max_length=max_length_tokens,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    top_p=top_p,
                    chunk_size=512
                ):
                    full_response += x

            # Clean up response
            response = full_response.strip()
            for stop_str in stop_words:
                response = response.replace(stop_str, "").strip()

            return response

        except Exception as e:
            raise Exception(f"Error during prediction: {str(e)}") 