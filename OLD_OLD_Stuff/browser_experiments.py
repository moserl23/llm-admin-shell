from openai import OpenAI
from playwright.sync_api import sync_playwright
import json, time
from config import API_KEY

client = OpenAI(api_key=API_KEY)

# -------------------------------------------------------------------
# Collect all visible, clickable elements (links & buttons)
# -------------------------------------------------------------------
def get_page_state(page):
    elements = []
    for link in page.locator("a").all()[:40]:
        try:
            name = link.inner_text().strip()
            if name:
                elements.append({"type": "link", "name": name})
        except:
            pass
    for btn in page.locator("button").all()[:20]:
        try:
            name = btn.inner_text().strip()
            if name:
                elements.append({"type": "button", "name": name})
        except:
            pass

    return {
        "title": page.title(),
        "url": page.url,
        "elements": elements
    }

# -------------------------------------------------------------------
# Ask GPT-4o to choose which visible element to click
# -------------------------------------------------------------------
def call_llm(goal, state, history):
    prompt = f"""
You are a web automation assistant.
Your goal: {goal}

Current page:
Title: {state['title']}
URL: {state['url']}

Clickable elements:
{json.dumps(state['elements'], ensure_ascii=False, indent=2)}

Action history:
{json.dumps(history, ensure_ascii=False, indent=2)}

Rules:
- If the current page title or URL already indicate the goal is reached, return {{"action":"finish","reason":"goal reached"}}.
- Otherwise, choose exactly one clickable element that moves toward the goal.
- Avoid repeating the same click multiple times.

Return ONLY one raw JSON object:
{{"action": "click", "name": "<exact visible text of element>"}}
or
{{"action": "finish", "reason": "<why>"}}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    text = response.choices[0].message.content.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print("⚠️ LLM returned non-JSON, ignoring.")
        print(text)
        return {"action": "finish", "reason": "invalid JSON"}


# -------------------------------------------------------------------
# Main automation routine
# -------------------------------------------------------------------
def main():
    goal = "Open the user menu and then go to 'Einstellungen' (settings). Finish when the settings page is open."


    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=500)
        context = browser.new_context(ignore_https_errors=True)
        page = context.new_page()

        # --- Login ---
        page.goto("https://nextcloud.local/")
        page.fill("#user", "admin")
        page.fill("#password", "changeme")
        page.click("button[data-login-form-submit]")
        page.wait_for_load_state("networkidle")

        # --- LLM-guided loop ---
        history = []
        for step in range(1, 6):
            state = get_page_state(page)
            decision = call_llm(goal, state, history)
            print(f"Step {step}: {decision}")

            history.append(decision)

            if decision.get("action") == "finish":
                print("✅ Finished:", decision.get("reason", ""))
                break

            if decision.get("action") == "click" and decision.get("name"):
                try:
                    page.get_by_text(decision["name"], exact=True).click()
                except Exception as e:
                    print("⚠️ Click failed:", e)
                    history.append("⚠️ Click failed: " + str(e))

            page.wait_for_load_state("networkidle")
            time.sleep(1)

        # --- Wrap up ---
        page.screenshot(path="final.png")
        print("Screenshot saved as final.png")
        browser.close()

if __name__ == "__main__":
    main()
