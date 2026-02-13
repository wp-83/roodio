<div class="{{ $customClass }}" style="{{ $customStyle }}">
    <x-button
        wire:click="toggle"
        wire:loading.attr="disabled"
        mood="{{ ($isFollowing) ? 'grayscale' : $mood }}"
        content="{{ ($isFollowing) ? 'Following' : 'Follow' }}"
    />
</div>