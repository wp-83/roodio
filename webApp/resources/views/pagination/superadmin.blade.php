@if ($paginator->hasPages())
    <div class="flex gap-2">
        {{-- Previous Page Link --}}
        @if ($paginator->onFirstPage())
            <span class="px-3 py-1.5 rounded-lg border border-shadedOfGray-20 text-shadedOfGray-30 cursor-not-allowed text-sm">Prev</span>
        @else
            <a href="{{ $paginator->previousPageUrl() }}" class="px-3 py-1.5 rounded-lg border border-shadedOfGray-20 text-shadedOfGray-60 hover:bg-primary-10 text-sm transition-colors">Prev</a>
        @endif

        {{-- Pagination Elements --}}
        @foreach ($elements as $element)
            {{-- "Three Dots" Separator --}}
            @if (is_string($element))
                <span class="px-3 py-1.5 rounded-lg border border-shadedOfGray-20 text-shadedOfGray-60 text-sm">{{ $element }}</span>
            @endif

            {{-- Array Of Links --}}
            @if (is_array($element))
                @foreach ($element as $page => $url)
                    @if ($page == $paginator->currentPage())
                        {{-- ACTIVE STATE (Biru Gelap) --}}
                        <span class="px-3 py-1.5 rounded-lg bg-primary-100 text-white hover:bg-primary-85 text-sm transition-colors cursor-default">{{ $page }}</span>
                    @else
                        {{-- INACTIVE STATE (Border Abu) --}}
                        <a href="{{ $url }}" class="px-3 py-1.5 rounded-lg border border-shadedOfGray-20 text-shadedOfGray-60 hover:bg-primary-10 text-sm transition-colors">{{ $page }}</a>
                    @endif
                @endforeach
            @endif
        @endforeach

        {{-- Next Page Link --}}
        @if ($paginator->hasMorePages())
            <a href="{{ $paginator->nextPageUrl() }}" class="px-3 py-1.5 rounded-lg border border-shadedOfGray-20 text-shadedOfGray-60 hover:bg-primary-10 text-sm transition-colors">Next</a>
        @else
            <span class="px-3 py-1.5 rounded-lg border border-shadedOfGray-20 text-shadedOfGray-30 cursor-not-allowed text-sm">Next</span>
        @endif
    </div>
@endif
