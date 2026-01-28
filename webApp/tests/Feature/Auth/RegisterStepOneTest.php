<?php
namespace Tests\Feature\Auth;

use App\Models\Region;
use App\Models\User;
use App\Models\userDetails;
use Illuminate\Foundation\Testing\RefreshDatabase;
use Tests\TestCase;

class RegisterStepOneTest extends TestCase
{
    use RefreshDatabase;

    public function test_halaman_register_bisa_diakses()
    {
        $response = $this->get(route('register'));

        $response->assertStatus(200);
        $response->assertViewIs('auth.register');
    }

    public function test_user_bisa_submit_data_valid()
    {
        // FIX 1: Tambahkan 'continent' karena di database kolom ini NOT NULL
        $region = Region::create([
            'id'        => 'ID',
            'name'      => 'Indonesia',
            'continent' => 'Asia',
        ]);

        $dataInput = [
            'fullname' => 'John Doe',
            'email'    => 'johndoe@example.com',
            'dob'      => '2000-01-01',
            'gender'   => '1',
            'country'  => $region->id,
        ];

        $response = $this->post(route('auth.register'), $dataInput);

        $response->assertSessionHasNoErrors();
        $response->assertSessionHas('register.step1');
        $response->assertRedirect(route('register.validation'));
    }

    public function test_validasi_gagal_jika_field_kosong()
    {
        $response = $this->post(route('auth.register'), []);

        $response->assertSessionHasErrors(['fullname', 'email', 'dob', 'gender', 'country']);
    }

    public function test_validasi_gagal_jika_email_sudah_ada()
    {
        // FIX 2: Tambahkan 'username' saat membuat User Parent
        // Karena User Factory bawaan mungkin belum ada username-nya
        $user = User::factory()->create([
            'username' => 'existing_user',
            'password' => bcrypt('password123'),
        ]);

        // FIX 3: Tambahkan 'continent' juga disini
        $region = Region::create([
            'id'        => 'ID',
            'name'      => 'Indonesia',
            'continent' => 'Asia',
        ]);

        // Buat UserDetails yang terhubung
        userDetails::create([
            'userId'      => $user->id,
            'fullname'    => 'Existing User',
            'email'       => 'taken@example.com',
            'dateOfBirth' => '1990-01-01',
            'gender'      => 1,
            'countryId'   => $region->id,
        ]);

        // Act: Coba register dengan email yang sama ('taken@example.com')
        $response = $this->post(route('auth.register'), [
            'fullname' => 'New User',
            'email'    => 'taken@example.com', // Email Duplicate
            'dob'      => '2000-01-01',
            'gender'   => '1',
            'country'  => $region->id,
        ]);

        $response->assertSessionHasErrors('email');
    }
}
