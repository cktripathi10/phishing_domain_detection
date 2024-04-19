import sys
import traceback
import logging

class CustomException(Exception):
    """Custom Exception class for more detailed error handling and logging."""
    def __init__(self, error, error_detail=None):
        super().__init__(str(error))
        self.error_message = self.format_error_message(error, error_detail)
        # Log the error message when an exception is created
        logging.error(self.error_message)

    @staticmethod
    def format_error_message(error, error_detail):
        """
        Formats a detailed error message including the file name, line number, and error message.
        """
        if error_detail is not None:
            _, _, exc_tb = sys.exc_info()
            tb_trace = traceback.extract_tb(exc_tb)
            # Gets the last traceback tuple which typically includes relevant file and line number
            filename, line, func, text = tb_trace[-1]
            return f"Error occurred in script: [{filename}] at line [{line}] in function [{func}]: {error}"
        else:
            return str(error)

    def __str__(self):
        return self.error_message


# Example usage of CustomException
if __name__ == "__main__":
    try:
        # Simulate an error for demonstration
        x = 1 / 0
    except Exception as e:
        raise CustomException(e, sys.exc_info())
