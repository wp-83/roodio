<div>
    <x-button wire:click="toggle" mood="{{ ($isFollowing) ? 'grayscale' : $mood }}" content="{{ ($isFollowing) ? 'Following' : 'Follow' }}" style='zoom:0.8;'></x-button>
</div>
