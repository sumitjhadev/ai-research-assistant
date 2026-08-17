@@
-from backend.config import (
-    BACKOFF_BASE_SECONDS,
-    DEFAULT_TOP_K,
-    GEMINI_MODEL_NAME,
-    MAX_RETRIES,
-    get_logger,
-    require_google_api_key,
-)
+from backend.config import (
+    BACKOFF_BASE_SECONDS,
+    DEFAULT_TOP_K,
+    MAX_RETRIES,
+    get_logger,
+)
@@
-import google.generativeai as genai
+from backend.llm import generate
@@
-    answer = _generate(_QA_SYSTEM_PROMPT, user_prompt)
+    answer = generate(_QA_SYSTEM_PROMPT, user_prompt)
@@
-    summary = _generate(_SUMMARY_SYSTEM_PROMPT, user_prompt)
+    summary = generate(_SUMMARY_SYSTEM_PROMPT, user_prompt)
@@
-    comparison = _generate(_COMPARE_SYSTEM_PROMPT, user_prompt)
+    comparison = generate(_COMPARE_SYSTEM_PROMPT, user_prompt)
