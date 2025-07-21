from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto("https://www.wi-line.fr/login.php")

    # Remplir identifiants
    page.fill('input[name="login"]', 'VOTRE_EMAIL')
    page.fill('input[name="password"]', 'VOTRE_MDP')
    page.click('button[type="submit"]')

    # Aller à la page de transaction
    page.goto("https://www.wi-line.fr/historique.php")

    # Attendre que les données chargent
    page.wait_for_timeout(3000)

    # Extraire le tableau ou cliquer sur exporter
    page.screenshot(path="capture.png")

    browser.close()
