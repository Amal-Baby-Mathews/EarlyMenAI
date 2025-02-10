from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate
from example_chat import chat_examples
class FewShotChatAssistant:
    def __init__(self, top_k=3,chat_examples=chat_examples):
        """
        Initializes the few-shot assistant with given examples.

        :param examples: List of dictionaries with "input" and "output" keys.
        :param top_k: Number of closest examples to retrieve.
        """
        self.examples = chat_examples
        self.top_k = top_k
        self.vectorizer = TfidfVectorizer()  # TF-IDF vectorizer
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        """Precomputes TF-IDF vectors for stored examples."""
        self.example_texts = [ex["input"] for ex in self.examples]
        self.example_vectors = self.vectorizer.fit_transform(self.example_texts)

    def get_most_similar_examples(self, query):
        """
        Finds the most similar examples using cosine similarity.

        :param query: User's query.
        :return: List of closest examples.
        """
        query_vector = self.vectorizer.transform([query])  # Convert query to vector
        similarities = cosine_similarity(query_vector, self.example_vectors).flatten()
        top_indices = similarities.argsort()[-self.top_k:][::-1]  # Get top-k highest scores
        return [self.examples[i] for i in top_indices]

    def generate_prompt(self, query,fetched_data="",available_actions= ["create_fire", "pick_apple"],chat_history=[]):
        """
        Creates a few-shot prompt dynamically based on the query.

        :param query: User's input query.
        :return: Formatted prompt string.
        """
        similar_examples = self.get_most_similar_examples(query)

        # Define the example prompt structure
        example_prompt = ChatPromptTemplate.from_messages(
            [('human', '{input}'), ('ai', '{output}')]
        )

        # Create few-shot prompt with closest examples
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            examples=similar_examples,
            example_prompt=example_prompt,
        )

        # Construct final prompt
        final_prompt = ChatPromptTemplate.from_messages(
            [
                ('system', 'You are a Sarcastic AI Assistant. Respond with a creative unique short text only message which is sarcastic and funny. Punching down on the user is ok. But output the functions accurately.Select and compile a python list of functions to be done from the available functions given according to user prompts.These are example prompts>///'),
                few_shot_prompt,
                ('human', '///> User Input:{input} \n Available Functions:{available_actions} \n Chat history:{chat_history} \n Additional Context:{fetched_data} \n'),
            ]
        )

        return final_prompt.format(input=query,available_actions=available_actions, chat_history=chat_history, fetched_data=fetched_data)

# Instantiate the chat assistant
assistant = FewShotChatAssistant()

# Example query
query = "fire"
formatted_prompt = assistant.generate_prompt(query)

# Print generated prompt
print(formatted_prompt)