@if ($paginator->hasPages())
    <nav role="navigation" aria-label="{{ __('Pagination Navigation') }}" class="flex items-center justify-between">

        {{-- Mobile View (Simple Previous/Next) --}}
        <div class="flex justify-between flex-1 sm:hidden">
            @if ($paginator->onFirstPage())
                <span class="relative inline-flex items-center px-4 py-2 text-sm font-medium text-shadedOfGray-60 bg-primary-85 border border-primary-60 rounded-md cursor-default leading-5 opacity-50">
                    {!! __('pagination.previous') !!}
                </span>
            @else
                <a href="{{ $paginator->previousPageUrl() }}" class="relative inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-primary-85 border border-primary-60 rounded-md leading-5 hover:bg-primary-70 focus:outline-none transition duration-150 ease-in-out">
                    {!! __('pagination.previous') !!}
                </a>
            @endif

            @if ($paginator->hasMorePages())
                <a href="{{ $paginator->nextPageUrl() }}" class="relative inline-flex items-center px-4 py-2 ml-3 text-sm font-medium text-white bg-primary-85 border border-primary-60 rounded-md leading-5 hover:bg-primary-70 focus:outline-none transition duration-150 ease-in-out">
                    {!! __('pagination.next') !!}
                </a>
            @else
                <span class="relative inline-flex items-center px-4 py-2 ml-3 text-sm font-medium text-shadedOfGray-60 bg-primary-85 border border-primary-60 rounded-md cursor-default leading-5 opacity-50">
                    {!! __('pagination.next') !!}
                </span>
            @endif
        </div>

        {{-- Desktop View --}}
        <div class="hidden sm:flex-1 sm:flex sm:items-center sm:justify-between">

            {{-- Info Text (Showing 1 to 10 of 50 results) --}}
            <div>
                <p class="text-sm text-shadedOfGray-30 font-secondaryAndButton">
                    Showing
                    <span class="font-bold text-white">{{ $paginator->firstItem() }}</span>
                    to
                    <span class="font-bold text-white">{{ $paginator->lastItem() }}</span>
                    of
                    <span class="font-bold text-white">{{ $paginator->total() }}</span>
                    results
                </p>
            </div>

            {{-- Page Numbers --}}
            <div>
                <span class="relative z-0 inline-flex shadow-sm rounded-md gap-2">

                    {{-- Previous Page Link --}}
                    @if ($paginator->onFirstPage())
                        <span aria-disabled="true" aria-label="{{ __('pagination.previous') }}">
                            <span class="relative inline-flex items-center px-3 py-2 text-sm font-medium text-shadedOfGray-60 bg-primary-85 border border-primary-60 rounded-lg cursor-default leading-5" aria-hidden="true">
                                <i class="bi bi-chevron-left"></i>
                            </span>
                        </span>
                    @else
                        <a href="{{ $paginator->previousPageUrl() }}" rel="prev" class="relative inline-flex items-center px-3 py-2 text-sm font-medium text-shadedOfGray-30 bg-primary-85 border border-primary-60 rounded-lg leading-5 hover:text-white hover:bg-primary-70 hover:border-primary-50 focus:z-10 focus:outline-none transition duration-150 ease-in-out" aria-label="{{ __('pagination.previous') }}">
                            <i class="bi bi-chevron-left"></i>
                        </a>
                    @endif

                    {{-- Pagination Elements --}}
                    @foreach ($elements as $element)
                        {{-- "Three Dots" Separator --}}
                        @if (is_string($element))
                            <span aria-disabled="true">
                                <span class="relative inline-flex items-center px-4 py-2 text-sm font-medium text-shadedOfGray-60 bg-primary-85 border border-primary-60 cursor-default leading-5 rounded-lg">{{ $element }}</span>
                            </span>
                        @endif

                        {{-- Array Of Links --}}
                        @if (is_array($element))
                            @foreach ($element as $page => $url)
                                @if ($page == $paginator->currentPage())
                                    <span aria-current="page">
                                        {{-- ACTIVE STATE: Happy Orange --}}
                                        <span class="relative inline-flex items-center px-4 py-2 text-sm font-bold text-white bg-secondary-happy-100 border border-secondary-happy-100 cursor-default leading-5 rounded-lg shadow-md shadow-secondary-happy-100/20">
                                            {{ $page }}
                                        </span>
                                    </span>
                                @else
                                    <a href="{{ $url }}" class="relative inline-flex items-center px-4 py-2 text-sm font-medium text-shadedOfGray-30 bg-primary-85 border border-primary-60 leading-5 rounded-lg hover:text-white hover:bg-primary-70 hover:border-primary-50 focus:z-10 focus:outline-none transition duration-150 ease-in-out" aria-label="{{ __('Go to page :page', ['page' => $page]) }}">
                                        {{ $page }}
                                    </a>
                                @endif
                            @endforeach
                        @endif
                    @endforeach

                    {{-- Next Page Link --}}
                    @if ($paginator->hasMorePages())
                        <a href="{{ $paginator->nextPageUrl() }}" rel="next" class="relative inline-flex items-center px-3 py-2 text-sm font-medium text-shadedOfGray-30 bg-primary-85 border border-primary-60 rounded-lg leading-5 hover:text-white hover:bg-primary-70 hover:border-primary-50 focus:z-10 focus:outline-none transition duration-150 ease-in-out" aria-label="{{ __('pagination.next') }}">
                            <i class="bi bi-chevron-right"></i>
                        </a>
                    @else
                        <span aria-disabled="true" aria-label="{{ __('pagination.next') }}">
                            <span class="relative inline-flex items-center px-3 py-2 text-sm font-medium text-shadedOfGray-60 bg-primary-85 border border-primary-60 rounded-lg cursor-default leading-5" aria-hidden="true">
                                <i class="bi bi-chevron-right"></i>
                            </span>
                        </span>
                    @endif
                </span>
            </div>
        </div>
    </nav>
@endif
