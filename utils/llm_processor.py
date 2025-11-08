# Aaroh_project/utils/llm_processor.py
from google import genai
from pydantic import BaseModel, Field
from typing import List
import os

# --- Pydantic Schemas for Structured Output ---

class QuizItem(BaseModel):
    """Defines the schema for a single quiz question."""
    question: str = Field(description="A clear comprehension question based on the input text.")
    type: str = Field(description="The question format, must be 'multiple_choice' or 'short_answer'.")
    correct_answer: str = Field(description="The correct answer to the question.")

class AarohOutput(BaseModel):
    """The final structured output containing all required learning aids."""
    simplified_text: str = Field(description="The complex text rewritten using simple, accessible language for a high school student.")
    analogy: str = Field(description="A single, highly memorable, real-world analogy to make the core concept 'sticky'.")
    quiz_questions: List[QuizItem] = Field(description="A list of exactly 3 high-quality quiz questions.")


# --- LLM Processing Function ---

def get_aaroh_output(complex_text):
    """
    Calls the Gemini API to perform the three core tasks in one structured response.
    """
    # The client automatically picks up the GEMINI_API_KEY from the environment
    try:
        client = genai.Client()
    except Exception as e:
        # Handle case where key is missing or invalid
        return f"Initialization Error: Could not connect to Gemini API. Check your GEMINI_API_KEY. Details: {e}", False


    # 1. Craft the Prompt (The Project Aaroh persona)
    # Aaroh_project/utils/llm_processor.py (within the get_aaroh_output function)

# --- REWRITE THIS SECTION ---
    system_prompt = (
        "You are Project Aaroh, a friendly and extremely patient teacher. "
        "Your mission is to explain complex topics and phrases in a way that a five-year-old child "
        "can easily and instantly understand. Use simple words, short sentences, and common, "
        "everyday examples (like toys, food, or pets). Avoid all technical jargon. "
        "Your output must still be in the requested JSON structure: "
        "1. **Simple Explanation:** The ELI5 explanation for the topic."
        "2. **Analogy:** A single, perfect, and easy-to-visualize analogy for a child."
        "3. **Quiz:** Three simple questions a five-year-old could answer to check understanding."
        "Ensure the output strictly adheres to the provided JSON Schema."
    )
# --- END REWRITE ---


    # 2. Configure the API call for structured output
    config = genai.types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_schema=AarohOutput, # Enforce the Pydantic schema
        temperature=0.4 # Lower temperature favors accuracy over extreme creativity
    )
    
    # 3. Call the model
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash', # A great model for fast, structured text tasks
            contents=[complex_text],
            config=config,
        )

        # 4. Parse the guaranteed JSON response into the Pydantic model
        # The response.text is a JSON string adhering to the AarohOutput schema
        return AarohOutput.model_validate_json(response.text).model_dump(), True

    except Exception as e:
        # Catch any errors during the API call or parsing
        return f"LLM Processing Error: Failed to generate content. Details: {e}", False