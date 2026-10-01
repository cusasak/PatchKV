def get_query(task, q=None):
    if task == "repeat":
        query = "Repeat the previous context exactly."
    elif task == "qa":
        if q is None:
            query = "Q: Answer the question based on the previous context."
        else:
            query = f"Q: {q}"
    elif task == "reason":
        query = (
            "Reason and answer the question. You must say the answer in the last sentence "
            f"beginning with 'The answer is'. Q: {q}"
        )
    elif task == "summarize":
        query = "Please summarize the previous context."
    else:
        raise ValueError(f"Invalid task: {task}")

    return query


class DataWrapper:
    def __init__(self, dataname, dataset, model, enable_thinking=False):
        self.name, self.dataset, self.model = dataname, dataset, model
        model.set_chat_template(dataname, enable_thinking=enable_thinking)

    def __len__(self):
        return len(self.dataset)

    def queries(self, idx):
        task = "reason" if self.name == "gsm" else "qa"
        record = self.dataset[idx]
        if not record["question"]:
            raise ValueError(f"No questions in context {idx}")
        if len(record["question"]) != len(record["answers"]):
            raise ValueError(f"Question/answer count differs in context {idx}")
        return [self.model.apply_template(get_query(task, q)) for q in record["question"]]

    def generate_answers(self, queries, kv):
        return [self.model.generate(query, kv=kv) for query in queries]
