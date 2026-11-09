"""ASGI bridge for the standalone dashboard container; no model is loaded here."""

from a2wsgi import WSGIMiddleware

from src.frontend.app import server


app = WSGIMiddleware(server, workers=4)
