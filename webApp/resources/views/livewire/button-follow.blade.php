<div>
    <x-button wire:click="toggle" mood="{{ ($isFollowing) ? 'grayscale' : $mood }}" content="{{ ($isFollowing) ? 'Following' : 'Follow' }}" style='zoom:0.75;'></x-button>
</div>
