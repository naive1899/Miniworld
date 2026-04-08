#!/usr/bin/env python3
"""
Запуск всех 9 конфигураций (3 режима × 3 seeds).
Использует существующий train.py.
"""
import subprocess
import sys
import time

# Конфигурация экспериментов
MODES = ['baseline', 'static', 'progressive']
SEEDS = [0, 1, 2]
STEPS = 200_000  # Для диплома: 500_000
NUM_ENVS = 4


def run_experiment(mode, seed):
    """Запуск одного эксперимента."""
    cmd = [
        sys.executable, 'train.py',
        '--mode', mode,
        '--seed', str(seed),
        '--steps', str(STEPS),
        '--envs', str(NUM_ENVS),
        '--save-dir', './models/'
    ]
    
    print(f"\n{'='*70}")
    print(f"Запуск: {mode.upper()} (seed={seed})")
    print(f"Команда: {' '.join(cmd)}")
    print(f"{'='*70}")
    
    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start
    
    if result.returncode == 0:
        print(f"✅ Успешно завершено за {elapsed/60:.1f} минут")
        return True
    else:
        print(f"❌ Ошибка!")
        return False


def main():
    print(f"План экспериментов: {len(MODES)} режима × {len(SEEDS)} seeds = {len(MODES)*len(SEEDS)} моделей")
    print(f"Timesteps: {STEPS:,}")
    print("Начало через 5 секунд (Ctrl+C для отмены)...")
    time.sleep(5)
    
    results = []
    
    # Последовательный запуск (стабильнее для Windows)
    for mode in MODES:
        for seed in SEEDS:
            success = run_experiment(mode, seed)
            results.append((mode, seed, success))
            
            # Небольшая пауза между запусками
            if not (mode == MODES[-1] and seed == SEEDS[-1]):
                print("Пауза 10 сек перед следующим запуском...")
                time.sleep(10)
    
    # Итоговая сводка
    print(f"\n{'='*70}")
    print("ИТОГОВАЯ СВОДКА")
    print(f"{'='*70}")
    successful = sum(1 for _, _, s in results if s)
    print(f"Успешно: {successful}/{len(results)}")
    
    for mode, seed, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {mode:12s} seed={seed}")
    
    # Список файлов для оценки
    print(f"\nФайлы для оценки:")
    for mode in MODES:
        for seed in SEEDS:
            path = f"models/{mode}_seed{seed}/final.zip"
            print(f"  python evaluate.py {path} --output results_{mode}_{seed}.json")


if __name__ == "__main__":
    main()