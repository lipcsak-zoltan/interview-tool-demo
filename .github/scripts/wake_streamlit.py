"""Visit the Streamlit Community Cloud app so it doesn't go to sleep.

Community Cloud hibernates apps after ~12 hours without visitors. This script
opens the app in a headless browser, clicks the wake-up button if the app is
asleep, and waits until the app itself has loaded (the demo password prompt).

Exits non-zero if the app never loads, so the workflow run shows as failed.
"""

import os
import sys
import time

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

APP_URL = os.environ.get(
    "STREAMLIT_APP_URL",
    "https://interview-tool-demo-zoltan-lipcsak.streamlit.app/",
)
WAKE_BUTTON_TEXT = "get this app back up"
READY_TEXT = "Enter the demo password"
STARTUP_TIMEOUT_S = 300  # waking a sleeping app can take a few minutes
STAY_ON_PAGE_S = 45  # keep the session open so it registers as a visit
SCREENSHOT_PATH = "wake-screenshot.png"


def find_in_frames(page, text):
    """Return the first locator containing `text` in the page or any iframe."""
    for frame in page.frames:
        try:
            locator = frame.get_by_text(text, exact=False)
            if locator.count() > 0:
                return locator.first
        except Exception:
            continue
    return None


def click_wake_button(page):
    for frame in page.frames:
        try:
            button = frame.get_by_role("button", name=WAKE_BUTTON_TEXT, exact=False)
            if button.count() > 0:
                button.first.click()
                return True
        except Exception:
            continue
    return False


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        print(f"Opening {APP_URL}")
        page.goto(APP_URL, wait_until="domcontentloaded", timeout=60_000)

        woke = False
        deadline = time.time() + STARTUP_TIMEOUT_S
        while time.time() < deadline:
            if find_in_frames(page, READY_TEXT):
                break
            if not woke and click_wake_button(page):
                woke = True
                print("App was asleep - clicked the wake-up button.")
            time.sleep(5)
        else:
            page.screenshot(path=SCREENSHOT_PATH, full_page=True)
            print(f"App did not load within {STARTUP_TIMEOUT_S}s. Screenshot saved.")
            browser.close()
            sys.exit(1)

        print("Woke it up." if woke else "Already awake.")
        time.sleep(STAY_ON_PAGE_S)
        page.screenshot(path=SCREENSHOT_PATH, full_page=True)
        browser.close()


if __name__ == "__main__":
    try:
        main()
    except PlaywrightTimeout as exc:
        print(f"Timed out: {exc}")
        sys.exit(1)
