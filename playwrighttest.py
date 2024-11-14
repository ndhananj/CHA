from playwright.sync_api import sync_playwright

def run(playwright):
    browser = playwright.chromium.launch()
    page = browser.new_page()
    page.goto("https://medium.com/@arjunattam/fast-and-reliable-cross-browser-testing-with-playwright-155c0e8a821f")
    print(page.title())
    browser.close()

with sync_playwright() as playwright:
    run(playwright)
