<div class="{{ $customClass }}">
    <x-button
        wire:click="toggle"
        wire:loading.attr="disabled"
        mood="{{ ($isFollowing) ? 'grayscale' : $mood }}"
        content="{{ ($isFollowing) ? 'Following' : 'Follow' }}"
        style='zoom:0.75;'
    />
</div>
