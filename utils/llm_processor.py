from google import genai
from pydantic import BaseModel, Field
from typing import List
import os

# --- Pydantic Schemas for Structured Output (3 Classes) ---

class SimplifiedOutput(BaseModel):
    """Schema for the first generation step: ELI5 explanation and analogy."""
    simplified_text: str = Field(description="The topic explained using simple, accessible language for a five-year-old.")
    analogy: str = Field(description="A single, highly memorable, real-world analogy for a child.")

class QuizItem(BaseModel):
    """Schema for a single quiz question, now focused on MCQ."""
    question: str = Field(description="A comprehension question based on the simplified text.")
    # We remove 'type' because it's now always multiple_choice
    correct_answer: str = Field(description="The correct answer from the list of options.")
    options: List[str] = Field(description="A list of 4 possible answers for a multiple-choice question.")

class QuizOutput(BaseModel):
    """Schema for the second generation step: The quiz."""
    quiz_questions: List[QuizItem] = Field(description="A list of exactly 3 high-quality multiple-choice questions.")

# --- LLM Helper Functions (Step 1 and 2) ---

def generate_simplification(client, complex_text):
    """Generates the ELI5 text and analogy (Step 1)."""
    system_prompt = (
        "You are Project Aaroh, a friendly teacher. Your goal is to explain the user's input "
        "topic/phrase using simple words, short sentences, and everyday examples (ELI5 style). "
        "Your output MUST be a JSON object with a simplified explanation and a single, related analogy."
    )
    config = genai.types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_schema=SimplifiedOutput,
        temperature=0.4
    )

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[complex_text],
        config=config,
    )
    # Returns a SimplifiedOutput Pydantic object
    return SimplifiedOutput.model_validate_json(response.text)


# Aaroh_project/utils/llm_processor.py

# ... (after generate_simplification function)

def generate_quiz(client, simplified_text):
    """Generates 3 multiple-choice quiz questions based on the simplified_text context."""
    quiz_prompt = (
        f"Based ONLY on the simplified explanation provided below, generate exactly 3 **multiple-choice questions (MCQs)**. "
        f"Each question must have **4 options** and be simple enough for a five-year-old child to answer, testing the ELI5 concept. "
        f"Ensure the output strictly adheres to the QuizOutput JSON Schema. \n\n"
        f"Simplified Text for Quiz Context: \n---\n{simplified_text}"
    )

    config = genai.types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=QuizOutput,
        temperature=0.1
    )

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[quiz_prompt],
        config=config,
    )
    return QuizOutput.model_validate_json(response.text)

def get_aaroh_output(complex_text):
    """Orchestrates the two-step prompt chain and correctly converts output to dicts."""
    try:
        # Client initialization will automatically pick up the GEMINI_API_KEY
        client = genai.Client()
        
        # 1. GENERATE SIMPLIFICATION (Step 1)
        step1_result = generate_simplification(client, complex_text)
        
        # 2. GENERATE QUIZ (Step 2)
        step2_result = generate_quiz(client, step1_result.simplified_text)
        
        # 3. COMBINE RESULTS (The essential FIX)
        final_output = {
            # Convert Step 1 Pydantic object fields directly to dictionary values
            "simplified_text": step1_result.simplified_text,
            "analogy": step1_result.analogy,
            
            # FIX: Use .model_dump() on the QuizOutput object to convert the
            # list of QuizItem objects into JSON-serializable dictionaries.
            "quiz_questions": step2_result.model_dump()["quiz_questions"]
        }
        
        return final_output, True

    except Exception as e:
        # A simple string error message for Flask to return
        return f"LLM Processing Error: Failed to complete chain. Details: {e}", False