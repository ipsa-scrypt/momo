import logging
from gradio_client import Client


def make_prediction(client, query):
    try:
        result = client.predict(query, api_name="/predict")
        return result
    except Exception as e:
        logging.error(f"An error occurred during prediction: {e}")
        return None


def main():
    logging.basicConfig(level=logging.INFO)
    client = Client("% API_URL %")
    query = "% QUERY %"
    result = make_prediction(client, query)
    if result is not None:
        print("Generated response:")
        print(result)


if __name__ == "__main__":
    main()
