## Модель × промпт × семплинг

| Модель | Промпт | Семплинг (T / top_p) | Док. | Валидных | Исправлено / failed | p50, с | p95, с | $ / 1000 док. | Точность (golden) | Текст | Вывод | Без нормализации |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemini-3.1-flash-lite | v1 | t0 (0.0 / 1.0) | 30 | 100% | 0 / 0 | 3.1 | 23.2 | $0.31 | **87%** | 86% | 90% | 87% |
| gemini-3.1-flash-lite | v1 | t07 (0.7 / 0.9) | 30 | 100% | 0 / 0 | 3.3 | 22.0 | $0.31 | **88%** | 88% | 90% | 88% |
| gemini-3.1-flash-lite | v2 | t0 (0.0 / 1.0) | 30 | 100% | 0 / 0 | 8.5 | 33.7 | $0.44 | **98%** | 98% | 97% | 97% |
| gemini-3.1-flash-lite | v2 | t07 (0.7 / 0.9) | 30 | 100% | 0 / 0 | 5.1 | 7.0 | $0.44 | **97%** | 97% | 97% | 95% |
| qwen2.5:3b | v1 | t0 (0.0 / 1.0) | 10 | 100% | 0 / 0 | 53.9 | 129.8 | $0.00 | **74%** | 72% | 80% | 66% |
| qwen2.5:3b | v1 | t07 (0.7 / 0.9) | 10 | 100% | 0 / 0 | 52.2 | 126.1 | $0.00 | **72%** | 71% | 73% | 62% |
| qwen2.5:3b | v2 | t0 (0.0 / 1.0) | 10 | 100% | 1 / 0 | 68.5 | 156.6 | $0.00 | **62%** | 64% | 53% | 58% |
| qwen2.5:3b | v2 | t07 (0.7 / 0.9) | 10 | 100% | 1 / 0 | 59.9 | 153.9 | $0.00 | **69%** | 70% | 67% | 63% |

## Точность по полям (golden set, 10 объявлений)

| Поле | gemini_v1_t0 | gemini_v1_t07 | gemini_v2_t0 | gemini_v2_t07 | ollama_v1_t0 | ollama_v1_t07 | ollama_v2_t0 | ollama_v2_t07 |
|---|---|---|---|---|---|---|---|---|
| rooms | 100% | 100% | 100% | 100% | 80% | 80% | 90% | 90% |
| area_m2 | 100% | 100% | 100% | 100% | 100% | 100% | 90% | 90% |
| price.amount | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| price.currency | 100% | 100% | 100% | 100% | 80% | 80% | 80% | 80% |
| price.utilities | 80% | 80% | 90% | 80% | 60% | 60% | 40% | 50% |
| location.city | 60% | 60% | 100% | 100% | 80% | 80% | 60% | 70% |
| location.district | 60% | 80% | 100% | 100% | 60% | 50% | 40% | 60% |
| pets_allowed | 100% | 100% | 90% | 90% | 90% | 90% | 80% | 80% |
| amenities | 70% | 70% | 100% | 100% | 0% | 0% | 0% | 10% |
| property_type | 100% | 100% | 100% | 100% | 100% | 90% | 100% | 100% |
| landlord_type | 80% | 80% | 100% | 100% | 50% | 40% | 40% | 70% |
| text_language | 90% | 90% | 90% | 90% | 90% | 90% | 20% | 30% |
| amenities precision / recall | 100% / 76% | 100% / 81% | 100% / 100% | 100% / 100% | 34% / 67% | 36% / 43% | 44% / 81% | 47% / 86% |

## Ошибки на golden set (id · поле · получено → ожидалось)

### gemini_v1_t0

- 001 · amenities · `['furniture']` → `['furniture', 'appliances', 'internet']`
- 001 · landlord_type · `unknown` → `agent`
- 002 · location.city · `Київ` → `None`
- 002 · location.district · `Подільський` → `None`
- 008 · price.utilities · `separate` → `unknown`
- 008 · location.city · `Київ` → `None`
- 008 · location.district · `Голосіївський` → `None`
- 008 · amenities · `['furniture']` → `['furniture', 'appliances', 'air_conditioner']`
- 011 · price.utilities · `separate` → `partial`
- 011 · amenities · `['balcony', 'boiler', 'air_conditioner']` → `['air_conditioner', 'boiler', 'internet', 'balcony']`
- 011 · landlord_type · `unknown` → `agent`
- 011 · text_language · `uk` → `mixed`
- 013 · location.district · `Замарстинівська` → `None`
- 023 · location.city · `Київ` → `None`
- 027 · location.city · `Київ` → `None`
- 027 · location.district · `Дніпровський` → `None`

### gemini_v1_t07

- 001 · amenities · `['furniture']` → `['furniture', 'appliances', 'internet']`
- 001 · landlord_type · `unknown` → `agent`
- 002 · location.city · `Київ` → `None`
- 002 · location.district · `Шевченківський` → `None`
- 008 · price.utilities · `separate` → `unknown`
- 008 · location.city · `Київ` → `None`
- 008 · amenities · `['furniture', 'air_conditioner']` → `['furniture', 'appliances', 'air_conditioner']`
- 011 · price.utilities · `separate` → `partial`
- 011 · amenities · `['balcony', 'boiler', 'air_conditioner']` → `['air_conditioner', 'boiler', 'internet', 'balcony']`
- 011 · landlord_type · `unknown` → `agent`
- 011 · text_language · `uk` → `mixed`
- 013 · location.district · `Замарстинівська` → `None`
- 023 · location.city · `Київ` → `None`
- 027 · location.city · `Київ` → `None`

### gemini_v2_t0

- 002 · price.utilities · `separate` → `unknown`
- 011 · pets_allowed · `negotiable` → `yes`
- 011 · text_language · `uk` → `mixed`

### gemini_v2_t07

- 002 · price.utilities · `separate` → `unknown`
- 011 · price.utilities · `separate` → `partial`
- 011 · pets_allowed · `negotiable` → `yes`
- 011 · text_language · `uk` → `mixed`

### ollama_v1_t0

- 001 · price.utilities · `included` → `separate`
- 001 · amenities · `['washing_machine', 'air_conditioner', 'furniture', 'internet', 'balcony', 'parking', 'balcony', 'furniture']` → `['furniture', 'appliances', 'internet']`
- 001 · landlord_type · `owner` → `agent`
- 002 · location.city · `Parkivyi Kwartal` → `None`
- 002 · location.district · `Tiraspol'skaya, monolitno-karaskovyi bydinnok` → `None`
- 002 · amenities · `['balcony', 'furniture', 'washing_machine', 'air_conditioner', 'furniture', 'washing_machine', 'furniture', 'furniture']` → `['furniture', 'appliances', 'air_conditioner', 'boiler', 'washing_machine', 'internet', 'balcony']`
- 008 · rooms · `2` → `1`
- 008 · location.city · `Vasylkivska` → `None`
- 008 · amenities · `['furniture', 'appliances', 'washing_machine', 'air_conditioner', 'balcony', 'parking', 'furniture', 'balcony']` → `['furniture', 'appliances', 'air_conditioner']`
- 008 · landlord_type · `unknown` → `agent`
- 011 · price.utilities · `unknown` → `partial`
- 011 · location.district · `None` → `центр`
- 011 · pets_allowed · `unknown` → `yes`
- 011 · amenities · `['balcony', 'air_conditioner', 'balcony', 'balcony', 'parking', 'internet', 'parking']` → `['air_conditioner', 'boiler', 'internet', 'balcony']`
- 011 · landlord_type · `unknown` → `agent`
- 011 · text_language · `uk` → `mixed`
- 013 · amenities · `['parking', 'parking', 'parking', 'parking', 'parking', 'parking', 'parking', 'parking']` → `['furniture', 'appliances', 'parking']`
- 013 · landlord_type · `owner` → `unknown`
- 014 · price.utilities · `included` → `separate`
- 014 · amenities · `['air_conditioner', 'balcony', 'furniture', 'internet', 'parking', 'washing_machine']` → `[]`
- 017 · amenities · `['parking', 'parking', 'parking']` → `[]`
- 020 · price.utilities · `included` → `separate`
- 020 · location.district · `Pivnitska Salyvka` → `Північна Салтівка`
- 020 · amenities · `['furniture', 'parking', 'balcony', 'air_conditioner', 'furniture', 'furniture', 'furniture', 'furniture']` → `['furniture']`
- 023 · price.currency · `UAH` → `USD`
- 023 · location.district · `Dniprovs'kyi raion (Dnipro District)` → `Дніпровський`
- 023 · amenities · `['internet', 'parking', 'washing_machine', 'washing_machine', 'washing_machine', 'washing_machine', 'washing_machine', 'washing_machine']` → `[]`
- 027 · rooms · `6` → `1`
- 027 · price.currency · `UAH` → `USD`
- 027 · amenities · `['air_conditioner', 'balcony', 'furniture', 'internet', 'parking', 'washing_machine']` → `[]`
- 027 · landlord_type · `owner` → `unknown`

### ollama_v1_t07

- 001 · price.utilities · `included` → `separate`
- 001 · amenities · `['washing_machine', 'furniture', 'washing_machine', 'washing_machine', 'washing_machine', 'washing_machine', 'washing_machine', 'washing_machine']` → `['furniture', 'appliances', 'internet']`
- 001 · landlord_type · `owner` → `agent`
- 002 · rooms · `2` → `1`
- 002 · location.city · `Kvitiv` → `None`
- 002 · location.district · `Parkovyi kвartal` → `None`
- 002 · amenities · `['balcony', 'furniture', 'furniture', 'furniture', 'furniture', 'furniture', 'furniture', 'furniture']` → `['furniture', 'appliances', 'air_conditioner', 'boiler', 'washing_machine', 'internet', 'balcony']`
- 008 · rooms · `2` → `1`
- 008 · location.city · `Vasylkivska Piska` → `None`
- 008 · location.district · `Park Land` → `None`
- 008 · amenities · `['balcony', 'furniture', 'furniture', 'furniture', 'furniture', 'furniture', 'furniture', 'furniture']` → `['furniture', 'appliances', 'air_conditioner']`
- 008 · landlord_type · `unknown` → `agent`
- 011 · price.utilities · `unknown` → `partial`
- 011 · location.district · `None` → `центр`
- 011 · pets_allowed · `unknown` → `yes`
- 011 · amenities · `['balcony', 'air_conditioner', 'balcony', 'balcony', 'parking', 'internet', 'parking']` → `['air_conditioner', 'boiler', 'internet', 'balcony']`
- 011 · landlord_type · `unknown` → `agent`
- 011 · text_language · `uk` → `mixed`
- 013 · amenities · `['parking', 'parking', 'parking', 'parking', 'parking', 'parking']` → `['furniture', 'appliances', 'parking']`
- 013 · landlord_type · `owner` → `unknown`
- 014 · price.utilities · `included` → `separate`
- 014 · amenities · `['internet', 'parking', 'washing_machine', 'washing_machine']` → `[]`
- 017 · amenities · `['parking', 'furniture', 'parking', 'parking', 'parking', 'parking', 'parking', 'parking']` → `[]`
- 020 · price.utilities · `included` → `separate`
- 020 · location.district · `Pivnitska Salyvka` → `Північна Салтівка`
- 020 · amenities · `['furniture', 'furniture', 'parking', 'air_conditioner', 'balcony', 'internet', 'furniture']` → `['furniture']`
- 020 · landlord_type · `owner` → `unknown`
- 023 · price.currency · `UAH` → `USD`
- 023 · location.district · `Dniprovs'kyi raion (Dnipro district)` → `Дніпровський`
- 023 · amenities · `['internet', 'parking']` → `[]`
- 027 · price.currency · `UAH` → `USD`
- 027 · amenities · `['balcony', 'parking']` → `[]`
- 027 · property_type · `room` → `apartment`
- 027 · landlord_type · `owner` → `unknown`

### ollama_v2_t0

- 001 · price.utilities · `partial` → `separate`
- 001 · amenities · `['furniture', 'appliances', 'air_conditioner', 'boiler', 'washing_machine', 'internet', 'balcony', 'parking']` → `['furniture', 'appliances', 'internet']`
- 001 · text_language · `ru` → `uk`
- 002 · price.utilities · `partial` → `unknown`
- 002 · location.city · `Тираспольська` → `None`
- 002 · location.district · `квітів Парковий квартал` → `None`
- 002 · pets_allowed · `negotiable` → `no`
- 002 · amenities · `['balcony', 'furniture', 'washing_machine', 'internet', 'parking', 'air_conditioner', 'boiler', 'furniture']` → `['furniture', 'appliances', 'air_conditioner', 'boiler', 'washing_machine', 'internet', 'balcony']`
- 002 · landlord_type · `agent` → `unknown`
- 002 · text_language · `ru` → `uk`
- 008 · price.utilities · `partial` → `unknown`
- 008 · location.city · `Васильківська` → `None`
- 008 · location.district · `ЖК Парк Ленд` → `None`
- 008 · amenities · `['furniture', 'appliances', 'air_conditioner', 'balcony', 'internet']` → `['furniture', 'appliances', 'air_conditioner']`
- 008 · text_language · `ru` → `uk`
- 011 · price.utilities · `unknown` → `partial`
- 011 · pets_allowed · `unknown` → `yes`
- 011 · amenities · `['balcony', 'internet']` → `['air_conditioner', 'boiler', 'internet', 'balcony']`
- 011 · text_language · `ru` → `mixed`
- 013 · location.district · `вул. Замарстинівська` → `None`
- 013 · amenities · `['furniture', 'appliances', 'balcony', 'internet']` → `['furniture', 'appliances', 'parking']`
- 013 · landlord_type · `owner` → `unknown`
- 013 · text_language · `ru` → `uk`
- 014 · area_m2 · `7.0` → `None`
- 014 · price.utilities · `included` → `separate`
- 014 · amenities · `['internet']` → `[]`
- 017 · location.district · `вул. Академіка Крип'якевича` → `None`
- 017 · amenities · `['balcony', 'furniture', 'internet']` → `[]`
- 017 · landlord_type · `owner` → `unknown`
- 017 · text_language · `ru` → `uk`
- 020 · price.utilities · `partial` → `separate`
- 020 · amenities · `['furniture', 'appliances', 'balcony', 'internet']` → `['furniture']`
- 020 · landlord_type · `agent` → `unknown`
- 020 · text_language · `ru` → `uk`
- 023 · price.currency · `UAH` → `USD`
- 023 · location.city · `Дніпровський район` → `None`
- 023 · location.district · `Никольско-Слободская` → `Дніпровський`
- 023 · amenities · `['balcony', 'internet']` → `[]`
- 023 · landlord_type · `owner` → `unknown`
- 023 · text_language · `ru` → `uk`
- 027 · rooms · `2` → `1`
- 027 · price.currency · `UAH` → `USD`
- 027 · location.city · `Київ` → `None`
- 027 · location.district · `ЖК Русанівська Гавань` → `None`
- 027 · amenities · `['balcony', 'furniture', 'appliances']` → `[]`
- 027 · landlord_type · `owner` → `unknown`

### ollama_v2_t07

- 001 · price.utilities · `partial` → `separate`
- 001 · amenities · `['furniture', 'appliances', 'air_conditioner', 'boiler', 'washing_machine', 'internet', 'balcony', 'parking']` → `['furniture', 'appliances', 'internet']`
- 002 · location.city · `Тираспольська` → `None`
- 002 · location.district · `квітів Парковий квартал` → `None`
- 002 · pets_allowed · `unknown` → `no`
- 002 · amenities · `['balcony', 'parking', 'furniture', 'appliances', 'washing_machine', 'internet', 'air_conditioner', 'boiler']` → `['furniture', 'appliances', 'air_conditioner', 'boiler', 'washing_machine', 'internet', 'balcony']`
- 002 · text_language · `ru` → `uk`
- 008 · price.utilities · `partial` → `unknown`
- 008 · location.city · `Васильків` → `None`
- 008 · location.district · `м. Васильківська пішка` → `None`
- 008 · amenities · `['furniture', 'appliances', 'air_conditioner', 'balcony', 'washing_machine', 'internet']` → `['furniture', 'appliances', 'air_conditioner']`
- 008 · text_language · `ru` → `uk`
- 011 · price.utilities · `unknown` → `partial`
- 011 · pets_allowed · `unknown` → `yes`
- 011 · amenities · `['balcony', 'furniture', 'internet', 'washing_machine', 'air_conditioner', 'boiler']` → `['air_conditioner', 'boiler', 'internet', 'balcony']`
- 011 · text_language · `uk` → `mixed`
- 013 · location.district · `ул. Замарстинівська` → `None`
- 013 · amenities · `['internet']` → `['furniture', 'appliances', 'parking']`
- 013 · landlord_type · `owner` → `unknown`
- 013 · text_language · `ru` → `uk`
- 014 · area_m2 · `70.0` → `None`
- 014 · price.utilities · `included` → `separate`
- 014 · amenities · `['internet']` → `[]`
- 017 · text_language · `ru` → `uk`
- 020 · price.utilities · `partial` → `separate`
- 020 · amenities · `['furniture', 'appliances', 'balcony']` → `['furniture']`
- 020 · landlord_type · `agent` → `unknown`
- 020 · text_language · `ru` → `uk`
- 023 · price.currency · `UAH` → `USD`
- 023 · location.city · `Дніпро́вський` → `None`
- 023 · amenities · `['internet']` → `[]`
- 027 · rooms · `3` → `1`
- 027 · price.currency · `UAH` → `USD`
- 027 · location.district · `ЖК Русанівська Гавань` → `None`
- 027 · amenities · `['furniture', 'appliances', 'internet', 'balcony']` → `[]`
- 027 · landlord_type · `owner` → `unknown`
- 027 · text_language · `ru` → `uk`
