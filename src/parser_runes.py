import requests
import urllib3
import pandas as pd
from bs4 import BeautifulSoup
import re
import time
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class RunesDBParser:
    """
    Парсер для базы данных рунических надписей runesdb.eu
    """
    
    def __init__(self):
        self.base_url = "https://www.runesdb.eu"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })
        self.images_dir = Path("runic_images")
        self.images_dir.mkdir(exist_ok=True)
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'with_images': 0,
            'without_images': 0
        }
    
    def find_id_range(self, sample_size=50, step=100):
        """
        Определение диапазона существующих ID на сайте
        
        Параметры:
        - sample_size: количество точек для проверки
        - step: шаг между проверяемыми ID
        """
        print("🔍 Определяем диапазон доступных ID...")
        
        test_ids = list(range(1, sample_size * step, step))
        existing_ids = []
        
        for test_id in test_ids:
            url = f"{self.base_url}/find/{test_id}"
            try:
                response = self.session.get(url, verify=False, timeout=10)
                if response.status_code == 200 and len(response.text) > 1000:
                    existing_ids.append(test_id)
                    print(f"✓ ID {test_id} существует")
            except:
                pass
        
        if existing_ids:
            min_id = min(existing_ids)
            max_id = max(existing_ids)
            print(f"\n📊 Найденный диапазон: {min_id} - {max_id}")
            print(f"   Проверено: {len(test_ids)} точек")
            print(f"   Найдено: {len(existing_ids)} существующих ID")
            return min_id, max_id
        else:
            print("⚠️  Не удалось определить диапазон")
            return None, None
    
    def smart_sample(self, total_count=100, start_id=1, end_id=5000):
        """
        Умная выборка ID для проверки
        """
        if total_count >= (end_id - start_id):
            return list(range(start_id, end_id + 1))
        
        # Равномерная выборка по всему диапазону
        step = (end_id - start_id) // total_count
        return list(range(start_id, end_id, step))
    
    def parse_find_page(self, url, find_id=None):
        """
        Улучшенный парсинг страницы отдельной находки
        """
        try:
            response = self.session.get(url, verify=False, timeout=30)
            
            # Проверка на пустую страницу
            if len(response.text) < 500:
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Извлекаем ID из URL
            if find_id is None:
                id_match = re.search(r'/find/(\d+)', url) or re.search(r'/f/(\d+)', url)
                find_id = id_match.group(1) if id_match else 'unknown'
            
            data = {
                'find_id': str(find_id),
                'url': url,
                'images': [],
                'transliteration': None,
                'translation': None,
                'runerow': None,
                'findplace': None,
                'country': None,
                'object_class': None,
                'dating': None,
                'material': None,
                'dimensions': None,
                'inscription_location': None,
                'full_text': None
            }
            
            # Извлекаем ВСЕ изображения
            for img in soup.find_all('img'):
                img_src = img.get('src', '')
                # Ищем изображения из image-service
                if 'image-service' in img_src or 'upload' in img_src:
                    img_url = img_src if img_src.startswith('http') else f"{self.base_url}{img_src}"
                    # Преобразуем в full-size если это thumbnail
                    img_url = img_url.replace('/thumb/', '/full/').replace('/preview/', '/full/')
                    if img_url not in data['images']:
                        data['images'].append(img_url)
            
            # Также ищем ссылки на изображения в атрибутах data-* и href
            for elem in soup.find_all(attrs={'data-image': True}):
                img_url = elem['data-image']
                if not img_url.startswith('http'):
                    img_url = f"{self.base_url}{img_url}"
                if img_url not in data['images']:
                    data['images'].append(img_url)
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                if 'image-service' in href or (href.endswith(('.jpg', '.jpeg', '.png', '.gif'))):
                    img_url = href if href.startswith('http') else f"{self.base_url}{href}"
                    if img_url not in data['images']:
                        data['images'].append(img_url)
            
            # Извлекаем текстовые данные через структурированный поиск
            text = soup.get_text()
            
            # Улучшенные regex для извлечения данных
            patterns = {
                'transliteration': r'Transliteration[:\s]*\n*\s*([^\n]+)',
                'translation': r'Translation[:\s]*\n*\s*([^\n]+)',
                'runerow': r'Runerow[:\s]*\n*\s*([^\n]+)',
                'findplace': r'Find[\s-]?place[:\s]*\n*\s*([^\n]+)',
                'country': r'Country[:\s]*\n*\s*([^\n]+)',
                'object_class': r'Object[\s-]?class[:\s]*\n*\s*([^\n]+)',
                'dating': r'Dating[:\s]*\n*\s*([^\n]+)',
                'material': r'Material[:\s]*\n*\s*([^\n]+)',
                'dimensions': r'Dimensions[:\s]*\n*\s*([^\n]+)',
                'inscription_location': r'Inscription[\s-]?location[:\s]*\n*\s*([^\n]+)'
            }
            
            for field, pattern in patterns.items():
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    data[field] = match.group(1).strip()
            
            # Сохраняем весь текст для дополнительного анализа
            data['full_text'] = text[:2000]  # Первые 2000 символов
            
            return data
            
        except Exception as e:
            print(f"⚠️  Ошибка при парсинге {url}: {e}")
            return None
    
    def download_image(self, img_url, find_id, img_index):
        """
        Улучшенное скачивание изображений
        """
        try:
            response = self.session.get(img_url, verify=False, timeout=30, stream=True)
            if response.status_code == 200:
                # Определяем расширение из Content-Type
                content_type = response.headers.get('Content-Type', '')
                ext_map = {
                    'image/jpeg': 'jpg',
                    'image/png': 'png',
                    'image/gif': 'gif',
                    'image/webp': 'webp'
                }
                ext = ext_map.get(content_type, 'jpg')
                
                # Если не удалось определить, пробуем из URL
                if ext == 'jpg':
                    url_ext = img_url.split('.')[-1].split('?')[0].lower()
                    if url_ext in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
                        ext = url_ext
                
                filename = self.images_dir / f"{find_id}_{img_index}.{ext}"
                
                # Скачиваем по частям для больших файлов
                with open(filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                file_size = filename.stat().st_size
                print(f"   ✓ Сохранено: {filename.name} ({file_size // 1024} KB)")
                return str(filename)
        except Exception as e:
            print(f"   ✗ Ошибка загрузки {img_url}: {e}")
        return None
    
    def scrape_all(self, start_id=1, end_id=5000, delay=1, max_workers=5):
        """
        Массовое скачивание с многопоточностью
        
        Параметры:
        - start_id: начальный ID
        - end_id: конечный ID
        - delay: задержка между запросами (секунды)
        - max_workers: количество параллельных потоков
        """
        results = []
        total = end_id - start_id + 1
        
        print(f"\n🚀 Начинаем парсинг ID от {start_id} до {end_id}")
        print(f"   Всего: {total} записей")
        print(f"   Потоков: {max_workers}")
        print(f"   Задержка: {delay}с\n")
        
        def process_id(find_id):
            url = f"{self.base_url}/find/{find_id}"
            
            try:
                data = self.parse_find_page(url, find_id=find_id)
                
                if data and (data.get('transliteration') or data.get('images')):
                    # Скачиваем изображения
                    local_images = []
                    for idx, img_url in enumerate(data['images']):
                        local_path = self.download_image(img_url, find_id, idx)
                        if local_path:
                            local_images.append(local_path)
                        time.sleep(0.5)  # Небольшая задержка между изображениями
                    
                    data['local_images'] = local_images
                    
                    self.stats['successful'] += 1
                    if local_images:
                        self.stats['with_images'] += 1
                    else:
                        self.stats['without_images'] += 1
                    
                    print(f"✓ [{find_id}] Найдено: {len(data['images'])} изобр., "
                          f"загружено: {len(local_images)}")
                    return data
                else:
                    self.stats['failed'] += 1
                    return None
                    
            except Exception as e:
                self.stats['failed'] += 1
                return None
        
        # Последовательная обработка с задержкой
        for find_id in range(start_id, end_id + 1):
            self.stats['total_processed'] += 1
            
            if self.stats['total_processed'] % 50 == 0:
                self.print_stats()
            
            result = process_id(find_id)
            if result:
                results.append(result)
            
            time.sleep(delay)
        
        return results
    
    def print_stats(self):
        """Вывод статистики"""
        print(f"\n📊 Статистика:")
        print(f"   Обработано: {self.stats['total_processed']}")
        print(f"   Успешно: {self.stats['successful']}")
        print(f"   Неудачно: {self.stats['failed']}")
        print(f"   С изображениями: {self.stats['with_images']}")
        print(f"   Без изображений: {self.stats['without_images']}\n")
    
    def create_dataframe(self, results):
        """
        Создание DataFrame из результатов
        """
        if not results:
            print("⚠️  Нет данных для создания DataFrame")
            return pd.DataFrame()
        
        # Преобразуем списки изображений в строки
        for r in results:
            if 'images' in r:
                r['images_count'] = len(r['images'])
                r['images_urls'] = '; '.join(r['images'])
                del r['images']
            if 'local_images' in r:
                r['local_images_count'] = len(r['local_images'])
                r['local_images_paths'] = '; '.join(r['local_images'])
                del r['local_images']
        
        df = pd.DataFrame(results)
        
        # Сортируем колонки для удобства
        priority_cols = ['find_id', 'findplace', 'country', 'transliteration', 
                        'translation', 'runerow', 'dating', 'images_count', 'url']
        other_cols = [col for col in df.columns if col not in priority_cols]
        df = df[[col for col in priority_cols if col in df.columns] + other_cols]
        
        return df


# === ВАРИАНТЫ ИСПОЛЬЗОВАНИЯ ===

def mode_1_quick_test():
    """Быстрый тест на небольшом диапазоне"""
    parser = RunesDBParser()
    print("🧪 РЕЖИМ 1: Быстрый тест (10 записей)")
    results = parser.scrape_all(start_id=4780, end_id=4790, delay=1)
    return parser, results


def mode_2_smart_sampling():
    """Умная выборка по всему диапазону"""
    parser = RunesDBParser()
    print("🎯 РЕЖИМ 2: Умная выборка")
    
    # Определяем диапазон
    min_id, max_id = parser.find_id_range()
    if not min_id:
        min_id, max_id = 1, 5000
    
    # Берем выборку из 100 записей
    sample_ids = parser.smart_sample(total_count=100, start_id=min_id, end_id=max_id)
    print(f"Выбрано {len(sample_ids)} ID для проверки")
    
    results = []
    for idx, find_id in enumerate(sample_ids, 1):
        print(f"\n[{idx}/{len(sample_ids)}] ID: {find_id}")
        url = f"{parser.base_url}/find/{find_id}"
        data = parser.parse_find_page(url, find_id=find_id)
        if data:
            results.append(data)
        time.sleep(1)
    
    return parser, results


def mode_3_full_scrape():
    """Полное скачивание всей базы"""
    parser = RunesDBParser()
    print("🌍 РЕЖИМ 3: Полное скачивание базы")
    print("⚠️  ВНИМАНИЕ: Это может занять несколько часов!")
    
    confirm = input("Продолжить? (yes/no): ")
    if confirm.lower() != 'yes':
        return None, []
    
    results = parser.scrape_all(start_id=1, end_id=7850, delay=1)
    return parser, results


if __name__ == "__main__":
    print("╔═══════════════════════════════════╗")
    print("║   RunesDB Парсер v2.0             ║")
    print("╚═══════════════════════════════════╝\n")
    print("Выберите режим работы:")
    print("1 - Быстрый тест (ID 4780-4790)")
    print("2 - Умная выборка (100 записей)")
    print("3 - Полное скачивание (вся база)")
    
    mode = input("\nВаш выбор (1/2/3): ").strip()
    
    if mode == '1':
        parser, results = mode_1_quick_test()
    elif mode == '2':
        parser, results = mode_2_smart_sampling()
    elif mode == '3':
        parser, results = mode_3_full_scrape()
    else:
        print("❌ Неверный выбор")
        exit()
    
    # Создаем DataFrame и сохраняем
    if results:
        df = parser.create_dataframe(results)
        
        print("\n\n" + "="*50)
        print("📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
        print("="*50)
        parser.print_stats()
        
        print(f"\n📋 DataFrame:")
        print(f"   Строк: {len(df)}")
        print(f"   Колонок: {len(df.columns)}")
        print(f"\n   Колонки: {list(df.columns)}")
        
        if not df.empty:
            print(f"\n📝 Примеры данных:")
            print(df[['find_id', 'findplace', 'country', 'images_count']].head())
        
        # Сохраняем
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"runic_inscriptions_{timestamp}.csv"
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        # Также сохраняем в JSON для удобства
        json_file = output_file.replace('.csv', '.json')
        df.to_json(json_file, orient='records', force_ascii=False, indent=2)
        
        print(f"\n✅ Данные сохранены:")
        print(f"   CSV: {output_file}")
        print(f"   JSON: {json_file}")
        print(f"   Изображения: {parser.images_dir}/")
    else:
        print("\n❌ Не удалось получить данные")