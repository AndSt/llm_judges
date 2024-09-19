from openai import OpenAI


def judgement_inference(
    comparison,
    question,
    thesis,
    antithesis,
    client,
    model="gpt-3.5-turbo-1106",
    temperature: float = 0.7,
):

    messages = [
        {
            "role": "user",
            "content": f"""Question:
{question}

Answer A:
{thesis}
--------------
Answer B: 
{antithesis}
--------------

Compare both answers in detail and choose the answer which correctly answers the question.

Conclude with a JSON in Markdown format indicating your choice between answer A or B:
```json
{{
    "answer":  "B" or "A"
}}
```
""",
        }
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=2048,
        logprobs=True,
    )

    return_object = {
        "question": question,
        "thesis": thesis,
        "antithesis": antithesis,
        "messages": messages,
        "response": response.choices[0].message.content,
        "full_response": response,
        "idA": comparison["model_1_id"],
        "idB": comparison["model_2_id"],
    }
    return return_object


def xxx_judgement_inference(
    comparison,
    question,
    thesis,
    antithesis,
    client,
    model="gpt-3.5-turbo-1106",
    temperature: float = 0.7,
):

    messages = [
        {
            "role": "user",
            "content": f"""Question:
{question}

Answer A:
{thesis}
--------------
Answer B: 
{antithesis}
--------------

Compare both answers in detail and choose the answer which correctly answers the question. 
Only analyze the reasoning! Therefore we exchanged all numbers with 'X' so you can focus on the reasoning.

Conclude with a JSON in Markdown format indicating your choice between answer A or B:
```json
{{
    "answer":  "B" or "A"
}}
```
""",
        }
    ]

    # The placeholder 'X' is inserted for all numbers in the answer to anonymize the data. Only focus on the reasoning of the answer.
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=2048,
        logprobs=True,
    )

    return_object = {
        "question": question,
        "thesis": thesis,
        "antithesis": antithesis,
        "messages": messages,
        "response": response.choices[0].message.content,
        "full_response": response,
        "idA": comparison["model_1_id"],
        "idB": comparison["model_2_id"],
    }
    return return_object


def llama3_judgement_inference(
    comparison,
    question,
    thesis,
    antithesis,
    client,
    model="gpt-3.5-turbo-1106",
    temperature: float = 0.7,
):

    messages = [
        {
            "role": "user",
            "content": f"""Question:
{question}

Answer A:
{thesis}
--------------
Answer B: 
{antithesis}
--------------

Compare both answers in detail and choose the answer which correctly answers the question.

Conclude with a JSON in Markdown format indicating your choice between answer A or B:
```json
{{
    "answer":  "B" or "A"
}}
```
""",
        }
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        logprobs=True,
        max_tokens=2048,
        stop=[
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eot_id|>",
            "<|reserved_special_token|>",
        ],
    )

    return_object = {
        "question": question,
        "thesis": thesis,
        "antithesis": antithesis,
        "messages": messages,
        "response": response.choices[0].message.content,
        "full_response": response,
        "idA": comparison["model_1_id"],
        "idB": comparison["model_2_id"],
    }
    return return_object


def judgement_inference_wrapper(
    comparison_type,
    comparison,
    question,
    opinion_1,
    opinion_2,
    url=None,
    model="gpt-3.5-turbo-1106",
    temperature: float = 0.7,
):
    if url is not None:
        client = OpenAI(base_url=url, api_key="EMPTY")
    else:
        client = OpenAI()

    if comparison_type == "llama3":
        compare_answers_fct = llama3_judgement_inference
    elif comparison_type == "std_xxx":
        compare_answers_fct = xxx_judgement_inference
    else:
        compare_answers_fct = judgement_inference

    generation = compare_answers_fct(
        comparison,
        question,
        opinion_1,
        opinion_2,
        client,
        model=model,
        temperature=temperature,
    )

    client.close()
    return {"comparison": comparison, "generation": generation}
