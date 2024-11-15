import os
import sys

def error_message_detail(error, error_detail:sys):
    _,_, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message =  f"Error in file: {file_name}, Error Line: {exc_tb.tb_lineno}, 
                        Error Message: {str(error)}"

    return error_message


class USvisaException(Exception):
    def  __init__(self, error_message, error_detail):
        """
        This is a custom exception class for handling errors in the application.
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail)


    def __str__(self):
        return self.error_message