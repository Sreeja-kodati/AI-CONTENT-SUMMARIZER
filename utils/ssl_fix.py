"""Configure SSL so HTTPS calls to xAI/Groq work on Windows."""

_configured = False


def configure_ssl() -> None:
    global _configured
    if _configured:
        return
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        import certifi
        import os

        bundle = certifi.where()
        os.environ.setdefault("SSL_CERT_FILE", bundle)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", bundle)
    _configured = True
