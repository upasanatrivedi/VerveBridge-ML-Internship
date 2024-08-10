import logging

def setup_logging():
    logging.root.handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("app.log", mode='w'),
            logging.StreamHandler()
        ]
    )
    logging.root.handlers[0].setLevel(logging.INFO)
    logging.root.handlers[1].setLevel(logging.INFO)