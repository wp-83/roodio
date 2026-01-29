<?php
namespace Tests\Feature\Auth;

use App\Models\User;
use Illuminate\Foundation\Testing\RefreshDatabase;
use Tests\TestCase;

class LogoutTest extends TestCase
{
    use RefreshDatabase;

    public function test_user_bisa_logout()
    {
        $user = User::factory()->create([
            'username' => 'testuser',
            'password' => bcrypt('password'),
        ]);

        $this->actingAs($user);

        $response = $this->post(route('auth.logout'));

        $response->assertRedirect(route('login'));
        $this->assertGuest();
    }
}
