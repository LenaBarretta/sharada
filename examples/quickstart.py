"""Ask one text several questions, and see what the probabilities are worth.

    python examples/quickstart.py
"""

from sharada import DecisionModel, Request, latency

MODEL = "lenabarretta/sharada-base"

TICKET = ("My card still hasn't arrived and I ordered it two weeks ago. The tracking page has said "
          "'in transit' since Monday and nobody answers the chat. I'd like to cancel and get my money "
          "back if it isn't here by Friday.")

QUESTIONS = [
    Request(text=TICKET, question="Which team should handle this?",
            options=["billing", "card delivery", "technical support", "account closure"],
            kind="choice", task="routing"),
    Request(text=TICKET, question="How urgent is this?",
            options=["can wait", "this week", "today", "right now"],
            kind="scale", task="urgency"),
    Request(text=TICKET, question="Is the customer asking for money back?",
            options=["no", "yes"], kind="binary", task="refund"),
    Request(text=TICKET, question="How annoyed does the customer sound?",
            options=["calm", "mildly annoyed", "frustrated", "furious"],
            kind="scale", task="tone"),
]


def main() -> None:
    model = DecisionModel.from_pretrained(MODEL)

    for request, decision in zip(QUESTIONS, model.decide_many(QUESTIONS)):
        print(f"\n{request.question}")
        print(f"  {decision.answer}  ({decision.confidence:.2f})")
        for option, p in sorted(decision.probabilities.items(), key=lambda kv: -kv[1]):
            print(f"    {p:5.2f}  {option}")

    # The options are part of the request, so a question the model has never been trained on still
    # gets an answer — and the same text, read once, can be asked all of them.
    print("\na label set from nowhere:")
    print(model.decide(TICKET, "Which of these does this ticket most look like?",
                       ["a complaint about a product", "a complaint about a person",
                        "a question about how something works", "a request to be left alone"]))

    print("\nhow long one decision takes:", latency(model, QUESTIONS[0]))


if __name__ == "__main__":
    main()
