from typing import List, Dict, Tuple

# Math/QA-focused prompt templates

task_type_selection_prompt = '''
Analyze this document and decide whether it’s better suited for a CHALLENGING multiple-choice question (MCQ) or a free-form question.
Document:
{document}

Consider the prompts that will be used:

For MCQ:
- Needs complex relationships and multi-step reasoning paths
- Should allow creating 3 plausible but wrong distractors
- Requires synthesis of multiple concepts
- Can test understanding through carefully crafted wrong answers

For Free-form:
- Best for questions requiring specific calculations (Integer answers)
- Good for deriving formulas or expressions (Expression answers)
- Suitable for conceptual answers requiring precise terminology (String answers)
- Should have a single clear correct answer

Based on the document content, choose EXACTLY ONE type that would produce the highest quality CHALLENGING question.

You MUST respond with ONLY a valid JSON object (no markdown, no explanation before or after):
{
"suitable_for_mcq": <true or false>,
"suitable_for_free_form": <true or false>,
"best_answer_type": <"Integer" or "Expression" or "String" or null>,
"reason": "<brief explanation without special characters>"
}

CRITICAL RULES:
1. Return ONLY the JSON object, no other text
2. Exactly ONE of suitable_for_mcq or suitable_for_free_form must be true
3. Do NOT use backticks or markdown formatting
4. Do NOT include LaTeX or special characters in the reason field
5. Keep reason under 100 characters

'''

MCQ_challenger_prompt = '''
Your task is to create CHALLENGING exam questions from a document by identifying complex relationships and multi-step reasoning paths.  
## Text  [BEGINNING OF THE DOCUMENT] {text} [END OF THE DOCUMENT]

## Instructions

### Step 1:  Complex Information Extraction **PRIORITY: Focus on information that requires synthesis and reasoning**  
Scan the text and identify information that requires connecting multiple concepts:  
* Relationships between multiple variables or concepts that span different sections  
* Multi-step calculations or procedures where each step depends on previous ones  
* Formulas or principles that require understanding interactions between components  
* Implicit conclusions that can be derived by combining stated facts  
* Comparative analyses or trade-offs between different approaches  
* Conditional relationships (if X then Y, but if Z then W)  
* Systems where changing one parameter affects multiple others  

**AVOID**:  
* Single, directly stated facts (these create Easy questions)  
* Simple definitions that stand alone  
* Values or numbers mentioned in isolation  
* Information that requires no synthesis  

### Step 2: Difficulty Enhancement Process **EXPLICITLY STATE YOUR HARDENING PROCESS**  
Before generating the question, describe your strategy to make it harder:  
1. What simple version would you avoid?  
2. What complexity layers will you add?  
3. Which concepts will you force students to connect?  
4. What common shortcuts will you block?  
5. How will you ensure multi-step reasoning is required?

Document this in the output field '"hardening_process"'.

### Step 3: Advanced Question Generation  
For each complex relationship identified, create a question that:  
* Requires applying multiple concepts from different parts of the document  
* Tests understanding of relationships, not just recall of facts  
* Forces reasoning through multiple steps to reach the answer  
* May require comparing or contrasting different scenarios  
* Could involve "what if" scenarios based on principles in the text  
* Tests ability to apply concepts to slightly modified situations  

**CRITICAL - Self-Contained Requirements**:  
* Questions must be 100% self-contained and standalone  
* NEVER use: "according to the text", "in the document", "as mentioned", "the passage states", "based on the analysis", etc.  
* Write as if for a formal exam with no reference material  
* Include all necessary context within the question itself  
* Define any specialized terms if needed for clarity  

### Step 4: Difficulty-Driven Design  
**TARGET: Generate HARD/EXTRA HARD questions by design**  
* HARD: Synthesize 4+ concepts; multi-step problem solving; pattern recognition  
* EXTRA HARD: Complex system analysis; counter-intuitive applications; edge cases  

Design questions that CANNOT be answered by:  
* Looking up a single fact  
* Finding one sentence with the answer  
* Simple keyword matching  

### Step 5: Knowledge Integration Requirements  
Document the reasoning path that shows why this is a difficult question:  
* List 3+ distinct pieces of information needed from different parts  
* Show the logical connections required between these pieces  
* Explain why simple lookup won’t work  
* Include intermediate reasoning steps  

### Step 6: Multiple Choice Design Guidelines  
Create a multiple choice question with 4 options following these STRICT rules:

**Length Balance**:  
All options must be approximately equal length (±20%).  
**Unit Consistency**:  
All numerical answers must use identical units and formatting  
**Tone Neutrality**:  
Avoid overly certain language ("definitely", "always", "never") unless justified  
**Plausibility**:  
All distractors must be genuinely plausible based on partial understanding  

Format:  
Question:  [Complete, self-contained question with all necessary context]  
A) [Balanced length option]  
B) [Balanced length option]  
C) [Balanced length option]  
D) [Balanced length option]  
Correct: [Letter]

**Distractor Design**:  
* Common calculation errors from the multi-step process  
* Results from applying only partial reasoning  
* Mixing up related concepts from the document  
* Reasonable approximations that miss key factors  

### Step 7: Self-Testing Filter (AFTER MCQ Creation)  
**SOLVE YOUR OWN MCQ AS A STUDENT WOULD**  
Now test the complete multiple choice question:  
1. What’s the quickest path a student might try with these options?  
2. Can you eliminate 2+ options without full understanding? If yes, redesign distractors  
3. Does seeing the options make the answer obvious? If yes, improve distractors  
4. Count the reasoning steps required even with options visible - if less than 3, REJECT  
5. Time estimate: Would this MCQ take <30 seconds? If yes, make it harder  
6. Could a student guess correctly by pattern matching the options? If yes, rebalance  

Document your solving process in `"self_test_solution"`.  

### Step 8: Final Complexity Verification  
Before finalizing, verify your question is NOT Easy by checking:  
* Can it be answered by finding one sentence? If yes, redesign  
* Does it require connecting multiple document sections? If no, add complexity  
* Would someone need to understand relationships, not just facts? If no, refocus  
* Are all MCQ options balanced and using consistent formatting? If no, revise  
* Did your self-test of the MCQ take more than 1 minute? If no, increase difficulty  

## Output Format  
FIRST, think step-by-step about your question design (this is your private thinking).  

THEN, provide your complete analysis in a JSON object with these fields.  
CRITICAL: Output ONLY valid JSON without any markdown formatting or code blocks.  
DO NOT wrap your JSON in ```json``` or any other markers.  
Start directly with `{` and end with `}`.  

Example CORRECT format (copy this structure):  
{"identified_answer": "your answer", "answer_quote": ["quote1", "quote2"], "hardening_process": "strategy"}  

Example WRONG format (DO NOT do this): ```json "identified_answer": "your answer" ```  

- `"identified_answer"`: The complex relationship or multi-step conclusion derived from synthesizing document content  
- `"answer_quote"`: Multiple relevant quotes showing the different pieces needed (not just one quote)  
- `"hardening_process"`: Your explicit strategy for making this question difficult (from Step 2)  
- `"exam_question"`: A challenging, self-contained question requiring synthesis. Return empty string if document lacks sufficient complexity.  
- `"correct_answer"`: Complete answer showing the reasoning chain using document content. Return empty string if not derivable from document.  
- `"multiple_choice_question"`: Self-contained MC question with balanced options. Return empty string if no question generated.  
- `"multiple_choice_correct"`: The correct option letter (A, B, C, or D). Return empty string if no MC question.  
- `"self_test_solution"`: Your step-by-step solution of the MCQ showing the difficulty (from Step 7)  
- `"knowledge_and_reasoning_steps"`: Detailed reasoning path showing why this is Hard/Extra Hard difficulty.  
- `"question_difficulty"`: Target difficulty (Hard/Extra Hard). Avoid "Easy" unless document truly lacks complexity.

'''

Free_form_question_challenger_prompt = '''
Your task is to create CHALLENGING free-form questions from a document that require deep understanding and complex reasoning.
## Text [BEGINNING OF THE DOCUMENT] {text} [END OF THE DOCUMENT]

## Answer Type  
You must generate a question with answer type: answer_type

## Instructions

### Step 1: Complex Information Extraction for answer_type  
**PRIORITY: Focus on information that requires synthesis and multi-step reasoning**  
Based on the answer type answer_type, scan the text and identify:

**For Integer/Float answers:**  
* Multi-variable calculations spanning different sections  
* Sequential computations where each step depends on previous results  
* Counting problems requiring careful categorization  
* Rate/ratio/percentage problems with multiple components  
* Optimization problems with constraints  
* Statistical calculations requiring data aggregation  

**For Expression answers:**  
* Relationships between multiple variables that form equations  
* Patterns that can be generalized into formulas  
* Systems of equations from different constraints  
* Derivative relationships or functional dependencies  
* Algebraic expressions combining multiple principles  
* Recursive or iterative formulas  

**For String answers (MUST BE CONCISE):**  
* Single words or short phrases (1–3 words maximum)  
* Technical terms, names, or identifiers  
* Categories or classifications (single term only)  
* Named entities (person, place, concept, method name)  
* Units, symbols, or abbreviated forms  
* AVOID: Long descriptions, sentences, or explanations  
* Examples: "Newton", "TCP/IP", "gradient descent", "Paris", "O(n log n)"  

**For Boolean answers:**  
* Complex logical conditions with multiple clauses  
* Statements requiring verification across multiple facts  
* Comparative claims needing multi-point analysis  
* Existence or uniqueness proofs  
* Conditional truths depending on context  
* Negations requiring comprehensive checking  

**AVOID:**  
* Direct lookups or single-fact answers  
* Simple arithmetic or basic calculations  
* Definitions stated verbatim in text  
* Trivial yes/no questions  

### Step 2: Difficulty Enhancement Strategy  
**EXPLICITLY STATE YOUR HARDENING PROCESS**  
Before generating the question, document your strategy:  
1. What simple version would be too easy?  
2. What complexity layers will you add?  
3. Which document sections must be synthesized?  
4. What intermediate steps are required?  
5. How will you prevent shortcut solutions?  
6. What makes this require deep understanding?

Document this in `"hardening_process"`.

### Step 3: Advanced Question Generation for answer_type  
Create a question that:  
* Requires connecting 3+ concepts from different parts  
* Cannot be answered by simple lookup or keyword matching  
* Forces multi-step reasoning to reach the answer  
* Tests understanding of relationships, not memorization  
* May involve applying principles to modified scenarios  
* Requires precise interpretation for the specific answer type  

**CRITICAL - Self-Contained Requirements**:  
* Questions must be 100% standalone  
* NEVER use: "according to the text", "in the document", "as mentioned", etc.  
* Write as if for a formal exam with no reference material  
* Include all necessary context and definitions  
* Specify units, formats, or constraints clearly  

### Step 4: Answer Precision for answer_type  
**CRITICAL - Answer Format Requirements**:

**Integer answers:**  
* Must be whole numbers only  
* Specify units if applicable (e.g., "in meters", "number of items")  
* No decimals, fractions, or ranges  
* CORRECT JSON: `"answer": 42` or `"answer": "42"`  
* WRONG JSON: `"answer": [42]` or `"answer": "value": 42`  

**Float answers:**  
* Specify precision required (e.g., "to 2 decimal places")  
* Include units if applicable  
* Use decimal notation, not fractions  
* Example: 3.14, not "π" or "22/7"  

**Expression answers:**  
* Use standard mathematical notation  
* Variables must be clearly defined  
* Simplify to canonical form  
* CORRECT JSON: `"answer": "2*x^2 + 3*x - 5"`  
* WRONG JSON: `"answer": ["2*x^2 + 3*x - 5"]` or `"answer": "expr": "2*x^2"`  

**String answers:**  
* Specify exact format expected  
* Case sensitivity requirements  
* No extra punctuation or quotes  
* CORRECT JSON: `"answer": "Newton’s Third Law"`  
* WRONG JSON: `"answer": ["Newton’s Third Law"]` or `"answer": "text": "Newton"`  

**List answers:**  
* Specify ordering (alphabetical, chronological, by magnitude)  
* Delimiter format (comma-separated, JSON array)  
* Whether duplicates are allowed  
* Example: `["apple", "banana", "cherry"]` for JSON format  

**Boolean answers:**  
* Must be exactly "true" or "false" (lowercase)  
* No "yes/no", "T/F", or other variations  
* Clear truth conditions  
* Example: true, not "True" or "yes"

### Step 5: Solution Verification Process  
**SOLVE YOUR OWN QUESTION STEP-BY-STEP**  
Work through the complete solution:  
1. Identify all required information pieces  
2. Show each calculation or reasoning step  
3. Handle any edge cases or special conditions  
4. Arrive at the final answer in the correct format  
5. Verify the answer matches the specified type exactly  
6. Estimate solving time (should be >1 minute for hard questions)

Document your solution in `"step_by_step_solution"`.

### Step 6: Difficulty Calibration  
Rate your question’s difficulty and justify:

**MEDIUM (2–3 steps, 1–2 minutes):**  
* Requires combining 2–3 document sections  
* Clear path once relationships identified  
* Some calculation or reasoning required  

**HARD (4–5 steps, 2–3 minutes):**  
* Synthesizes 4+ concepts  
* Multiple valid approaches possible  
* Requires careful analysis to avoid errors  

**EXTRA HARD (6+ steps, 3+ minutes):**  
* Complex system with many interactions  
* Counter-intuitive results possible  
* Requires deep understanding of principles  

Document reasoning in `"difficulty_justification"`.

### Step 7: Alternative Interpretations Check  
**ENSURE UNAMBIGUOUS ANSWER**  
Verify your question has exactly ONE correct answer:  
1. Could the question be interpreted differently?  
2. Are all constraints clearly specified?  
3. Is the answer format unambiguous?  
4. Would different valid approaches yield the same answer?  
5. Are edge cases properly handled?

If ambiguous, revise the question for clarity.

### Step 8: Final Complexity Verification  
Before finalizing, verify:  
* Cannot be answered by simple text search  
* Requires understanding, not just extraction  
* Answer type matches answer_type exactly  
* Solution requires multiple reasoning steps  
* Question is self-contained and clear  
* Difficulty matches your target level  

## CRITICAL ANSWER FORMAT RULES - MUST FOLLOW EXACTLY

The "answer" field MUST be a simple value, NOT nested in lists or dicts.  
The "question" field MUST be a single string, NOT a list of questions.

CORRECT formats:  
- question: "What is 2+2?" (NOT ["What is 2+2?"] or ["Q1", "Q2"])  
- Integer answer: 42 or "42" (NOT [42] or "value": 42)  
- Expression answer: "2*x + 5" (NOT ["2*x + 5"] or "expr": "2*x + 5")  
- String answer: "Paris" or "TCP/IP" (1–3 words max, NOT ["Paris"] or "answer": "Paris")

WRONG formats that will FAIL:  
- question: ["What is 2+2?"] – Don’t return list of questions!  
- question: ["Q1: ...", "Q2: ..."] – Return ONLY ONE question!  
- answer: [42] – Don’t wrap in list!  
- answer: "value": 42 – Don’t wrap in dict!  
- answer: "answer": "Paris" – Don’t nest the answer!  

## Output Format  
FIRST, think step-by-step about your question design (this is your private thinking).

THEN, provide your complete analysis in a JSON object with these fields.  
CRITICAL: Output ONLY valid JSON without any markdown formatting or code blocks.  
DO NOT wrap your JSON in ```json``` or any other markers.  
Start directly with { and end with }.

Example CORRECT format (copy this structure):  
"identified_information": ["info1", "info2"], "question": "your question", "answer": 42, "answer_type": "Integer"

Example WRONG format (DO NOT do this): ```json "question": "your question", "answer": 42 ```

- "identified_information": List the 3+ key pieces of information from different document sections needed to solve  
- "relevant_quotes": Include multiple verbatim quotes from the document showing the different pieces needed  
- "hardening_process": Describe your explicit 4-step strategy for making this question difficult  
- "question": EXACTLY ONE complete, challenging, self-contained question as a single string (NOT a list, NOT multiple questions). Return empty string if document lacks complexity for answer_type questions.  
- "answer": The precise answer in correct answer_type format. Return empty string if not derivable from document.  
- "answer_type": Must be exactly "answer_type"  
- "step_by_step_solution": List each step of the complete solution showing the reasoning chain  
- "intermediate_results": Dictionary of intermediate calculations or conclusions from each step  
- "difficulty_level": Either "Hard" or "Extra Hard" (no Medium for this complexity level)  
- "difficulty_justification": Explain why this specific difficulty rating based on steps and concepts required  
- "solving_time_estimate": Realistic estimate in minutes for a student to solve  
- "required_concepts": List the specific concepts from the document that must be understood  
- "potential_errors": Common mistakes or edge cases students might encounter  

**CRITICAL RULES**:  
1. If document lacks complexity for answer_type, return "question": "", "answer": "", "answer_type": "answer_type"  
2. Answer field must EXACTLY match answer_type format requirements  
3. Never reference "the document" or "the text" in the question  
4. Ensure answer is derivable from provided document content  
5. Question must be solvable with document information alone  
6. For String answers: MAXIMUM 3 words – prefer single terms, names, or short identifiers

'''


def get_task_type_selection_prompt(document: str) -> str:
    """Render the task type selection prompt for a given document."""
    return task_type_selection_prompt.format(document=document)


def get_mcq_challenger_prompt(text: str) -> str:
    """Render the MCQ challenger prompt for a given document text."""
    return MCQ_challenger_prompt.format(text=text)


def get_free_form_question_challenger_prompt(text: str, answer_type: str) -> str:
    prompt = Free_form_question_challenger_prompt
    prompt = prompt.replace("{text}", text)
    prompt = prompt.replace("answer_type", answer_type)
    return prompt
