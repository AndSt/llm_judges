from openai import OpenAI


def generate_few_shot_answer(
    idx, question, answer, result, model, temperature, url, few_shots=[]
):
    if url is not None:
        client = OpenAI(base_url=url, api_key="EMPTY")
    else:
        client = OpenAI()

    try:
        prompt = question

        messages = []
        is_first = True
        for shot in few_shots:
            if is_first:
                shot["question"] = (
                    f"You are a reasoning assistant. Always answer exactly in the same format. Use '####' to separate the final answer (without additional comments) from the reasoning.\n\n{shot['question']}"
                )
                is_first = False
            messages.extend(
                [
                    {"role": "user", "content": shot["question"]},
                    {"role": "assistant", "content": shot["answer"]},
                ]
            )

        # messages[-1]["content"] = messages[-1]["content"] + "\n\nYou are a reasoning assistant. Always answer exactly in the same format, using '####' to separate the final answer from the reasoning."

        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            # logprobs=True,
            max_tokens=1400,
        )

        outcome = {
            "idx": idx,
            "question": question,
            "result": result,
            "answer": answer,
            "prompt": prompt,
            "messages": messages,
            "response": response.choices[0].message.content,
            "full_response": response,
        }
    except Exception as e:
        print(e)
        outcome = {
            "idx": idx,
            "question": question,
            "result": result,
            "answer": answer,
            "prompt": prompt,
            "messages": messages,
            "response": None,
        }

    client.close()
    return outcome


def generate_few_shot_answer_llama3(
    idx, question, answer, result, model, temperature, url, few_shots=[]
):
    if url is not None:
        client = OpenAI(base_url=url, api_key="EMPTY")
    else:
        client = OpenAI()

    try:
        prompt = question

        messages = [
            {
                "role": "system",
                "content": "You are a reasoning assistant. Always answer exactly in the same format. Use '####' to separate the final answer (without additional comments) from the reasoning.",
            }
        ]
        for shot in few_shots:
            messages.extend(
                [
                    {"role": "user", "content": shot["question"]},
                    {"role": "assistant", "content": shot["answer"]},
                ]
            )

        messages.append({"role": "user", "content": prompt})

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

        outcome = {
            "idx": idx,
            "question": question,
            "result": result,
            "answer": answer,
            "prompt": prompt,
            "messages": messages,
            "response": response.choices[0].message.content,
            "full_response": response,
        }
    except Exception as e:
        print(e)
        outcome = {
            "idx": idx,
            "question": question,
            "result": result,
            "answer": answer,
            "prompt": prompt,
            "messages": messages,
            "response": None,
        }

    client.close()
    return outcome
