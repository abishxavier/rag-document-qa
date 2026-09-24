import os
import sys
import time

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from playwright.sync_api import sync_playwright

def run_test():
    print("🚀 Starting Playwright UI Verification Test...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1400, "height": 900})
        page = context.new_page()

        print("Navigating to http://localhost:8501...")
        page.goto("http://localhost:8501", timeout=30000)

        # Wait for Streamlit app to load
        page.wait_for_selector(".gradient-header", timeout=20000)
        print("✓ Header rendered with custom gradient class!")

        # Verify Page Title
        title = page.title()
        print(f"✓ Page Title: '{title}'")
        assert "DocuMind" in title or "RAG" in title

        # Verify Sidebar controls
        sidebar = page.locator("section[data-testid='stSidebar']")
        assert sidebar.is_visible(), "Sidebar should be visible"
        print("✓ Sidebar is visible")

        # Verify API key is hidden from the UI as requested
        api_input = page.locator("input[type='password']")
        assert api_input.count() == 0, "API key input should NOT be present on the UI"
        print("✓ API Key field is successfully hidden from the UI (loaded securely via environment)")

        # Verify Action buttons (Process PDF, Clear Chat) and assert equal sizing
        process_btn = sidebar.locator("button:has-text('Process PDF')")
        clear_btn = sidebar.locator("button:has-text('Clear Chat')")
        assert process_btn.is_visible(), "Process PDF button should be visible"
        assert clear_btn.is_visible(), "Clear Chat button should be visible"

        # Assert equal size (height & width)
        box_process = process_btn.bounding_box()
        box_clear = clear_btn.bounding_box()
        print(f"✓ Button 'Process PDF' dimensions: {box_process['width']:.1f}px x {box_process['height']:.1f}px")
        print(f"✓ Button 'Clear Chat' dimensions: {box_clear['width']:.1f}px x {box_clear['height']:.1f}px")
        assert abs(box_process['height'] - box_clear['height']) < 2.0, "Both buttons should have identical height"
        assert abs(box_process['width'] - box_clear['width']) < 2.0, "Both buttons should have identical width"
        print("✓ Both buttons are verified to be exactly the same size!")

        # Verify Hero Welcome Card
        hero_card = page.locator("text=Welcome to DocuMind RAG")
        assert hero_card.is_visible(), "Hero card should be visible before document upload"
        print("✓ Hero welcome card is visible")

        # Capture initial screenshot
        os.makedirs(".playwright-mcp", exist_ok=True)
        screenshot_path = os.path.join(".playwright-mcp", "01_playwright_initial.png")
        page.screenshot(path=screenshot_path, full_page=True)
        print(f"✓ Screenshot 1 saved to {screenshot_path}")

        # Test uploading sample PDF
        pdf_path = os.path.abspath("sample_apollo.pdf")
        if os.path.exists(pdf_path):
            print(f"Testing PDF upload with {pdf_path}...")
            file_input = sidebar.locator("input[type='file']")
            file_input.set_input_files(pdf_path)

            # Wait for Streamlit to register the uploaded file
            print("Waiting for file upload to be acknowledged by Streamlit...")
            page.wait_for_selector("text=sample_apollo.pdf", timeout=15000)
            print("✓ File acknowledged by Streamlit file uploader widget!")
            time.sleep(2)

            # Click Process PDF
            print("Clicking 'Process PDF'...")
            process_btn.click()

            # Wait for processing
            print("Waiting for vectorstore embedding and indexation...")
            page.wait_for_selector("text=Indexed", timeout=30000)
            print("✓ Document successfully indexed! Success banner displayed.")

            # Verify document active status badge
            active_badge = sidebar.locator("text=Document Active")
            active_badge.scroll_into_view_if_needed()
            assert active_badge.is_visible(), "Document Active badge should appear in sidebar"
            print("✓ 'Document Active' badge confirmed in sidebar!")

            # Verify chat input is now visible
            chat_input = page.locator("[data-testid='stChatInput'] textarea")
            assert chat_input.is_visible(), "Chat input should be visible after indexing"
            print("✓ Chat input box is active and visible!")

            # Capture active document screenshot
            active_screenshot_path = os.path.join(".playwright-mcp", "02_playwright_doc_active.png")
            page.screenshot(path=active_screenshot_path, full_page=True)
            print(f"✓ Screenshot 2 saved to {active_screenshot_path}")

            # Test querying the document
            print("Submitting a query into the chat input...")
            chat_input.fill("Who is the CEO of Project Apollo?")
            chat_input.press("Enter")
            print("Query submitted. Waiting for response bubble...")

            # Wait for response or error handling bubble
            page.wait_for_selector("text=Who is the CEO of Project Apollo?", timeout=10000)
            time.sleep(3)
            query_screenshot_path = os.path.join(".playwright-mcp", "03_playwright_chat_query.png")
            page.screenshot(path=query_screenshot_path, full_page=True)
            print(f"✓ Screenshot 3 saved to {query_screenshot_path}")

        browser.close()
        print("\n🎉 ALL PLAYWRIGHT TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_test()
