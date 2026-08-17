@@
-# Load environment variables from .env if present. Real secrets must never
-# be committed; only .env.example (with placeholders) is tracked in git.
-load_dotenv()
+# Load environment variables from .env if present. Real secrets must never
+# be committed; only .env.example (with placeholders) is tracked in git.
+load_dotenv()
@@
-GEMINI_MODEL_NAME = "gemini-2.5-flash"
+GEMINI_MODEL_NAME = "gemini-2.5-flash"
+
+# If MOCK_LLM is true, the code will use an internal deterministic mock LLM
+# which is safe to run in public demos without an API key. Set MOCK_LLM=false
+# in your deployment environment to use the real Gemini API (requires
+# GOOGLE_API_KEY to be set as an environment variable / secret).
+MOCK_LLM = os.environ.get("MOCK_LLM", "true").lower() in ("1", "true", "yes")
@@
-def require_google_api_key() -> str:
-    """Return the configured Gemini API key or raise a clear, actionable error.
-
-    Returns:
-        The value of the GOOGLE_API_KEY environment variable.
-
-    Raises:
-        RuntimeError: If GOOGLE_API_KEY is not set. This is intentionally a
-            clean error message (not a raw stack trace from the SDK) so a
-            missing key fails fast and obviously in local dev, CI, and prod.
-    """
-    if not GOOGLE_API_KEY:
-        raise RuntimeError(
-            "GOOGLE_API_KEY is not set. Create a .env file (see .env.example) "
-            "and set GOOGLE_API_KEY=<your key>, or export it in your shell. "
-            "Get a free key at https://aistudio.google.com/app/apikey"
-        )
-    return GOOGLE_API_KEY
+def require_google_api_key() -> str:
+    """Return the configured Gemini API key or raise a clear, actionable error.
+
+    If MOCK_LLM is enabled, this function will not raise an error so the
+    application can run in demo mode without a real key. When MOCK_LLM is
+    disabled, the function requires GOOGLE_API_KEY to be set and will raise
+    a RuntimeError with instructions if it's missing.
+    """
+    if not GOOGLE_API_KEY and not MOCK_LLM:
+        raise RuntimeError(
+            "GOOGLE_API_KEY is not set. Create a .env file (see .env.example) "
+            "and set GOOGLE_API_KEY=<your key>, or export it in your shell. "
+            "Get a free key at https://aistudio.google.com/app/apikey"
+        )
+    return GOOGLE_API_KEY
