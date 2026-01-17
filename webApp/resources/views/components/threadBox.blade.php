@props([
    'creator',
    'createdAt',
    'title',
    'content',
    'threadId'
])

<div class='bg-secondary-angry-10 rounded-lg w-sm h-max p-5 md:w-md lg:w-2xl'>
    <div class='flex flex-row items-center gap-5 w-full'>
        <div class='w-16 h-16 bg-primary-10 rounded-full flex items-center justify-center relative z-5 overflow-hidden'>
            {{-- @if (isset($user->userDetail->profilePhoto))
                <img src="{{ config('filesystems.disks.azure.url') . '/' . $user->userDetail->profilePhoto }}" alt="{{ $user->userDetail->fullname }}" class='w-full h-full object-cover'> 
            @else
                <p class='text-paragraph font-primary font-bold text-primary-70 h-fit'>{{ Str::charAt(Str::upper($user->userDetail->fullname), 0) }}</p>
                @endif --}}
            <p class='text-subtitle font-primary font-bold text-primary-70 h-fit'>{{ Str::charAt(Str::upper('William'), 0) }}</p>
        </div>
        <div class='flex-1 w-full'>
            <div class='flex flex-row justify-between'>
                <p>{{ $creator }}</p>
                <p>Follow</p>
            </div>
            <div>
                <p>{{ $createdAt }}</p>
            </div>
        </div>
    </div>
    <hr class='my-3'>
    <div>
        <p class='font-bold'>{{ $title }}</p>
        <p>{{ $content }}</p>
    </div>
    <div class='flex flex-row '>
        <livewire:user.reaction-button :thread-id="$threadId" />
    </div>
</div>