<div x-data="{ showPhotoModal: false }">
    <div class="max-w-4xl w-full mx-auto bg-white rounded-3xl shadow-2xl overflow-hidden relative mb-10">

        <div class="h-48 bg-gradient-to-r from-primary-100 to-primary-60 relative">
            <a href="{{ url()->previous() }}" class="absolute top-8 left-8 flex items-center gap-2 px-5 py-2 bg-white/10 backdrop-blur-md border border-white/20 rounded-full text-white hover:bg-white/20 transition group z-20">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 group-hover:-translate-x-1 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                </svg>
                <span class="text-smallBtn font-medium">Back</span>
            </a>
            <div class="absolute bottom-0 right-0 w-80 h-80 bg-accent-100 opacity-10 rounded-full translate-x-1/3 translate-y-1/3 blur-3xl pointer-events-none"></div>
            <div class="absolute top-0 right-20 w-40 h-40 bg-secondary-happy-100 opacity-10 rounded-full blur-2xl pointer-events-none"></div>
        </div>

        <div class="px-8 pb-12">
            <div class="flex flex-col md:flex-row justify-between items-end -mt-20 mb-10 gap-6 relative z-10">
                <div class="relative">
                    <div class="relative">
                        @if ($photo)
                            <img class="w-40 h-40 rounded-full border-[6px] border-white object-cover shadow-lg bg-shadedOfGray-20"
                                src="{{ $photo->temporaryUrl() }}"
                                alt="Profile Photo Preview">
                        @else
                            <img class="w-40 h-40 rounded-full border-[6px] border-white object-cover shadow-lg bg-shadedOfGray-20"
                                src="{{ config('filesystems.disks.azure.url') . '/' . $profilePhoto }}"
                                alt="Profile Photo">
                        @endif

                        <button type="button" @click="showPhotoModal = true" class="absolute bottom-0 right-0 bg-primary-85 text-white p-2.5 rounded-full border-4 border-white hover:bg-primary-100 transition shadow-md group" title="Change Photo">
                            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 group-hover:scale-110 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                            </svg>
                        </button>
                    </div>
                </div>

               <div class="w-full md:w-auto">
                    <button
                        wire:click="update"
                        disabled
                        wire:dirty.attr.remove="disabled"
                        wire:target="fullname,email,dateOfBirth,gender,countryId"
                        class="w-full md:w-auto px-8 py-3 rounded-xl font-medium text-mediumBtn text-white flex items-center justify-center gap-2 transition shadow-lg
                               bg-primary-60 hover:bg-primary-50 shadow-primary-20/50
                               disabled:opacity-50 disabled:cursor-not-allowed disabled:bg-shadedOfGray-30 disabled:shadow-none"
                    >
                        Save Changes
                    </button>
                </div>
            </div>

            <form wire:submit.prevent="update">
                <div class="grid grid-cols-1 lg:grid-cols-3 gap-12">
                    <div class="lg:col-span-2 space-y-8">
                        <div>
                            <h2 class="font-primary text-title text-primary-100 mb-2">Personal Information</h2>
                            <p class="text-shadedOfGray-60 text-small mb-8">Manage your personal details and public profile.</p>

                            <div class="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-8">
                                <div class="col-span-1 md:col-span-2">
                                    <label for="fullname" class="block text-smallBtn font-medium text-shadedOfGray-70 mb-2">Full Name</label>
                                    <input type="text" wire:model="fullname" id="fullname" name="fullname" class="w-full px-5 py-4 text-body-size border border-shadedOfGray-20 rounded-xl focus:border-primary-50 focus:ring-4 focus:ring-primary-10/40 outline-none transition text-shadedOfGray-100 placeholder-shadedOfGray-30 bg-white">
                                    @error('fullname') <span class="text-error-moderate">{{ $message }}</span> @enderror
                                </div>
                                <div class="col-span-1 md:col-span-2">
                                    <label for="email" class="block text-smallBtn font-medium text-shadedOfGray-70 mb-2">Email Address</label>
                                    <input type="email" wire:model="email" id="email" name="email" class="w-full px-5 py-4 text-body-size border border-shadedOfGray-20 rounded-xl focus:border-primary-50 focus:ring-4 focus:ring-primary-10/40 outline-none transition text-shadedOfGray-100 bg-white">
                                    @error('email') <span class="text-error-moderate">{{ $message }}</span> @enderror
                                </div>
                                <div>
                                    <label for="dob" class="block text-smallBtn font-medium text-shadedOfGray-70 mb-2">Date of Birth</label>
                                    <input type="date" wire:model="dateOfBirth" id="dob" name="dob" class="w-full px-5 py-4 text-body-size border border-shadedOfGray-20 rounded-xl focus:border-primary-50 focus:ring-4 focus:ring-primary-10/40 outline-none transition text-shadedOfGray-100 bg-white font-secondaryAndButton">
                                    @error('dateOfBirth') <span class="text-error-moderate">{{ $message }}</span> @enderror
                                </div>
                                <div>
                                    <label for="region" class="block text-smallBtn font-medium text-shadedOfGray-70 mb-2">Region</label>
                                    <div class="relative">
                                        <select wire:model="countryId" id="region" name="region" class="w-full px-5 py-4 text-body-size border border-shadedOfGray-20 rounded-xl focus:border-primary-50 focus:ring-4 focus:ring-primary-10/40 outline-none transition appearance-none bg-white text-shadedOfGray-100 cursor-pointer">
                                            @forelse($regions as $region)
                                                <option value="{{ $region->id }}" {{ $countryId === $region->id ? 'selected' : '' }}>{{ $region->name }}</option>
                                            @empty @endforelse
                                        </select>
                                        <div class="absolute inset-y-0 right-0 flex items-center px-4 pointer-events-none">
                                            <svg class="w-5 h-5 text-shadedOfGray-50" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-span-1 md:col-span-2">
                                    <label class="block text-smallBtn font-medium text-shadedOfGray-70 mb-2">
                                        Gender
                                    </label>

                                    <div class="flex gap-4">

                                        <label class="cursor-pointer w-full group">
                                            <input
                                                type="radio"
                                                name="gender"
                                                wire:model="gender"
                                                value="1"
                                                class="peer sr-only"
                                            >
                                            <div
                                                class="w-full p-4 border border-shadedOfGray-20 rounded-xl
                                                    flex items-center justify-center gap-3
                                                    peer-checked:border-primary-60
                                                    peer-checked:bg-primary-10/20
                                                    peer-checked:text-primary-85
                                                    transition group-hover:bg-shadedOfGray-10">
                                                <span class="text-body-size">Male</span>
                                            </div>
                                        </label>

                                        <label class="cursor-pointer w-full group">
                                            <input
                                                type="radio"
                                                name="gender"
                                                wire:model="gender"
                                                value="0"
                                                class="peer sr-only"
                                            >
                                            <div
                                                class="w-full p-4 border border-shadedOfGray-20 rounded-xl
                                                    flex items-center justify-center gap-3
                                                    peer-checked:border-primary-60
                                                    peer-checked:bg-primary-10/20
                                                    peer-checked:text-primary-85
                                                    transition group-hover:bg-shadedOfGray-10">
                                                <span class="text-body-size">Female</span>
                                            </div>
                                        </label>

                                    </div>

                                    @error('gender')
                                        <span class="text-error-moderate">{{ $message }}</span>
                                    @enderror
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="lg:col-span-1">
                        <div class="bg-shadedOfGray-10 rounded-2xl p-8 h-full border border-shadedOfGray-20">
                            <div>
                                <div class="flex items-center gap-3 mb-8">
                                    <div class="p-2.5 bg-white border border-shadedOfGray-20 rounded-xl text-primary-100 shadow-sm">
                                        <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                                        </svg>
                                    </div>
                                    <h3 class="font-primary text-body-size font-bold text-shadedOfGray-100">Account Security</h3>
                                </div>
                                <div class="space-y-6">
                                    <div>
                                        <label class="block text-micro font-bold text-shadedOfGray-60 mb-2 uppercase tracking-wider">Username</label>
                                        <div class="flex items-center bg-white border border-shadedOfGray-20 rounded-xl px-4 py-3">
                                            <span class="text-shadedOfGray-50 mr-2 font-medium">@</span>
                                            <input type="text" wire:model="username" class="w-full outline-none text-body-size text-shadedOfGray-85 font-medium bg-transparent" readonly>
                                        </div>
                                    </div>
                                    <div>
                                        <label class="block text-micro font-bold text-shadedOfGray-60 mb-2 uppercase tracking-wider">Password</label>
                                        <div class="relative">
                                            <input type="password" value="DummyPass123" class="w-full bg-white px-4 py-3 text-body-size border border-shadedOfGray-20 rounded-xl text-shadedOfGray-85 outline-none pr-20" readonly>
                                            <button type="button" class="absolute right-2 top-2 bottom-2 px-3 text-smallBtn text-accent-100 hover:bg-accent-10 rounded-lg font-medium transition">
                                                Change
                                            </button>
                                        </div>
                                    </div>
                                    <div class="pt-4">
                                        <p class="text-micro text-shadedOfGray-50 italic">Last password changed: 3 months ago</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </form>
        </div>

        <div x-show="showPhotoModal"
             x-cloak
             style="display: none;"
             class="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm transition-opacity duration-300">

            <div class="bg-white rounded-2xl w-full max-w-md mx-4 shadow-2xl overflow-hidden transform transition-all scale-100">

                <div class="bg-gradient-to-r from-primary-100 to-primary-60 p-6">
                    <h3 class="font-primary text-subtitle text-white">Update Photo</h3>
                    <p class="text-primary-10 text-small">Select a new image for your profile.</p>
                </div>

                @if (session('success'))
                <div class="flex items-start sm:items-center p-4 mb-4 text-sm text-fg-success-strong rounded-base bg-success-soft m-5" role="alert">
                    <svg class="w-4 h-4 me-2 shrink-0 mt-0.5 sm:mt-0" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" viewBox="0 0 24 24"><path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 11h2v5m-2 0h4m-2.592-8.5h.01M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z"/></svg>
                    <p>{{ session('success')}}</p>
                </div>
                @endif

                <div class="p-6">
                    @error('profilePhoto')
                    <div class="mb-4 p-3 rounded-lg bg-secondary-angry-10 border border-secondary-angry-20 text-secondary-angry-100 text-smallBtn flex items-start gap-2">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 flex-shrink-0" viewBox="0 0 20 20" fill="currentColor">
                            <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
                        </svg>
                        <span>{{ $message }}</span>
                    </div>
                    @enderror

                    <form wire:submit.prevent="uploadPhoto" enctype="multipart/form-data">
                        <div class="space-y-6">
                            <div class="flex justify-center">
                                <div class="relative w-40 h-40">
                                    @if ($photo)
                                        <img src="{{ $photo->temporaryUrl() }}" class="w-full h-full rounded-full object-cover border-4 border-shadedOfGray-20 bg-shadedOfGray-10">
                                    @else
                                        <img src="{{ config('filesystems.disks.azure.url') . '/' . $profilePhoto }}" class="w-full h-full rounded-full object-cover border-4 border-shadedOfGray-20 bg-shadedOfGray-10">
                                    @endif

                                    <label for="photoInputLivewire" class="absolute inset-0 flex items-center justify-center bg-black/40 rounded-full opacity-0 hover:opacity-100 transition cursor-pointer">
                                        <span class="text-white font-medium text-smallBtn">Click to Select</span>
                                    </label>
                                </div>
                            </div>

                            <input type="file" id="photoInputLivewire" wire:model="photo" class="hidden" name="photo">

                            <div class="text-center">
                                <button type="button" onclick="document.getElementById('photoInputLivewire').click()" class="text-primary-50 font-medium text-smallBtn hover:underline">
                                    Choose File
                                </button>
                                <p class="text-micro text-shadedOfGray-50 mt-1">JPG, PNG or GIF. Max 5MB.</p>

                                <div wire:loading wire:target="photo" class="text-secondary-happy-100 text-micro mt-2 font-medium">
                                    Uploading...
                                </div>
                            </div>

                            <div class="flex gap-3 pt-2">
                                <button type="button" @click="showPhotoModal = false" class="flex-1 py-3 border border-shadedOfGray-20 rounded-xl text-shadedOfGray-70 font-medium hover:bg-shadedOfGray-10 transition">
                                    Cancel
                                </button>
                                <button type="submit" class="flex-1 py-3 bg-primary-60 hover:bg-primary-50 text-white rounded-xl font-medium shadow-md shadow-primary-20/50 transition">
                                    Upload
                                </button>
                            </div>
                        </div>
                    </form>
                </div>
            </div>
        </div>

    </div>
</div>
