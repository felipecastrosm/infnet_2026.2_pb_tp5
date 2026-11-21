"""Captura telas do painel rodando com o modelo real (para a documentação e o PDF da entrega).

Abre o app no Chrome sem janela (Selenium), roda o agente consultor, faz uma pergunta de
acompanhamento no chat com memória e salva uma imagem de cada aba em docs/evidencias/telas/.

Uso (com a API e o app no ar):
    python scripts/capturar_telas.py
    SOLAR_APP_URL=http://localhost:8521 python scripts/capturar_telas.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.support.ui import WebDriverWait

URL = os.environ.get("SOLAR_APP_URL", "http://localhost:8501")
SAIDA = Path(__file__).resolve().parents[1] / "docs" / "evidencias" / "telas"
LARGURA, ALTURA = 1440, 5200


def esperar_texto(driver, texto: str, limite: int = 180) -> None:
    WebDriverWait(driver, limite).until(lambda d: texto in d.find_element(By.TAG_NAME, "body").text)


def clicar_botao(driver, texto: str) -> None:
    botao = WebDriverWait(driver, 60).until(
        ec.element_to_be_clickable((By.XPATH, f"//button[.//p[normalize-space()='{texto}']]"))
    )
    botao.click()


def aba(driver, texto: str) -> None:
    driver.find_element(By.XPATH, f"//button[@role='tab'][.//p[contains(normalize-space(), '{texto}')]]").click()
    time.sleep(3)


def salvar_aba(driver, nome: str) -> None:
    painel = driver.find_element(By.CSS_SELECTOR, "div[role='tabpanel']:not([hidden])")
    painel.screenshot(str(SAIDA / nome))
    print("salvo", nome)


def main() -> None:
    SAIDA.mkdir(parents=True, exist_ok=True)
    opcoes = webdriver.ChromeOptions()
    opcoes.add_argument("--headless=new")
    opcoes.add_argument(f"--window-size={LARGURA},{ALTURA}")
    opcoes.add_argument("--force-device-scale-factor=1")
    driver = webdriver.Chrome(options=opcoes)
    try:
        driver.get(URL)
        esperar_texto(driver, "Analisar com o agente")
        time.sleep(3)
        driver.set_window_size(LARGURA, 900)
        driver.save_screenshot(str(SAIDA / "01_painel_inicio.png"))
        print("salvo 01_painel_inicio.png")
        driver.set_window_size(LARGURA, ALTURA)

        clicar_botao(driver, "Analisar com o agente")
        esperar_texto(driver, "Como o agente decidiu")
        time.sleep(4)
        for expansor in driver.find_elements(By.XPATH, "//summary[.//p[contains(., 'modelo,')]]")[:2]:
            expansor.click()  # abre dois passos para mostrar as probabilidades
        time.sleep(1)
        salvar_aba(driver, "02_consultor_agente.png")

        campo = driver.find_element(By.CSS_SELECTOR, "textarea[placeholder='Pergunte sobre a recomendação']")
        for pergunta in ["Quanto o sistema gera no pior mês?", "E em Fortaleza, como seria?"]:
            campo = driver.find_element(By.CSS_SELECTOR, "textarea[placeholder='Pergunte sobre a recomendação']")
            campo.send_keys(pergunta, Keys.ENTER)
            esperar_texto(driver, pergunta)
            time.sleep(2)
            WebDriverWait(driver, 120).until(lambda d: "Consultando a memória" not in d.find_element(By.TAG_NAME, "body").text)
            time.sleep(3)
        driver.find_element(By.XPATH, "//summary[.//p[contains(., 'O que o assistente lembra')]]").click()
        time.sleep(1)
        chat = driver.find_element(By.XPATH, "//h3[contains(., 'Converse sobre o resultado')]/ancestor::div[contains(@class, 'stVerticalBlock')][1]")
        chat.screenshot(str(SAIDA / "03_chat_com_memoria.png"))
        print("salvo 03_chat_com_memoria.png")

        aba(driver, "Panorama das capitais")
        salvar_aba(driver, "04_panorama_capitais.png")
        aba(driver, "Setor solar com IA")
        salvar_aba(driver, "05_setor_com_ia.png")
        aba(driver, "Como funciona")
        salvar_aba(driver, "06_ciclo_ciencia_de_dados.png")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
