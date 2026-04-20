from config import REBRICKABLE_API_KEY, TOP_K_CANDIDATES, TOP_K_RESULTS
from search import search_rebrickable, rank_with_clip, display_results


def main():
    print("=" * 50)
    print("  레고 블록 검색기 (텍스트 → 이미지)")
    print("=" * 50)

    while True:
        query = input("\n찾고 싶은 블록을 설명하세요 (종료: q)\n> ").strip()
        if query.lower() == "q":
            break
        if not query:
            continue

        print(f'\n"{query}" 검색 중...')

        # 1단계: Rebrickable API 키워드 검색
        candidates = search_rebrickable(query, REBRICKABLE_API_KEY, TOP_K_CANDIDATES)
        if not candidates:
            print("검색 결과가 없습니다. 다른 키워드를 시도해보세요.")
            continue

        # 2단계: CLIP 유사도 재정렬
        print("\nCLIP으로 이미지 유사도 계산 중...")
        top_results = rank_with_clip(query, candidates, TOP_K_RESULTS)

        # 3단계: 결과 출력
        print("\n── 최종 결과 ──")
        for i, part in enumerate(top_results, 1):
            print(f"  {i}. {part['name']} (#{part['part_num']}) | 유사도: {part['score']*100:.1f}%")

        display_results(top_results, query)


if __name__ == "__main__":
    main()