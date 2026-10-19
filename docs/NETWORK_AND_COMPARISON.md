### Transport and A/B comparison (2026-10-20)

- Image uploads retry **once** after a browser connection failure, HTTP 502/504 or 429. The model inference endpoint is read-only; no storage or purchase action is repeated. Live camera has **no retries**, avoiding duplicated frames and runaway requests.
- Network failures now say whether the phone is offline or the connection was interrupted. The upload screen keeps an inline error message and avoids stacking identical popup notifications.
- Click **Compare Round 2 and Round 3** on the upload page to send the *same* source image and confidence threshold **sequentially** to both models. Results show box overlays and top confidence labels for each.
- More detections or higher confidence **does not** prove greater accuracy. Quality needs an agreed labeled validation collection evaluated separately.
- Neither model nor dataset is retrained or modified by these transport/UI changes.
